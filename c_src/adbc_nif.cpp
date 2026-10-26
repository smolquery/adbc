#include <arrow-adbc/adbc.h>
#include <cstdbool>
#include <cstdio>
#include <fine.hpp>
#include <memory>
#include <mutex>
#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow_ipc.h>
#include <nanoarrow/nanoarrow_ipc.hpp>
#include <stdexcept>

#include "from_arrow.hpp"
#include "releaser.hpp"
#include "shared.hpp"
#include "to_arrow.hpp"

namespace adbc_nif {

FINE_RESOURCE(ArrowArrayStreamRecord);

// Fine resource wrappers
//
// A resource's ADBC handle lives in an owner that the resource shares with
// the resources that depend on it: a connection's owner holds its
// database's owner, a statement's owner its connection's, and a stream
// holds the owners it reads through. The owner releases the handle when the
// last of them lets go, so a statement is released before its connection,
// and a connection before its database. The resource's `value` refers to
// the handle inside its owner, so the handle keeps its address for as long
// as anything holds it.
//
// Owners are let go on a `Releaser` worker, never on a BEAM scheduler,
// because a driver may block in a release (see releaser.hpp). The worker
// that lets go of the last owner of a chain releases the whole chain in the
// same job, without waiting on the VM to run another resource's destructor.

static void release_error(struct AdbcError *adbc_error) {
  if (adbc_error->release != nullptr) {
    adbc_error->release(adbc_error);
  }
}

template <typename Owner> static void let_go(std::shared_ptr<Owner> owner) {
  Releaser::run([owner = std::move(owner)]() mutable { owner.reset(); });
}

struct DatabaseOwner {
  struct AdbcDatabase value{};

  ~DatabaseOwner() {
    if (value.private_data != nullptr) {
      struct AdbcError adbc_error{};
      AdbcDatabaseRelease(&value, &adbc_error);
      release_error(&adbc_error);
    }
  }
};

struct ConnectionOwner {
  struct AdbcConnection value{};
  std::shared_ptr<DatabaseOwner> database;

  ~ConnectionOwner() {
    if (value.private_data != nullptr) {
      struct AdbcError adbc_error{};
      AdbcConnectionRelease(&value, &adbc_error);
      release_error(&adbc_error);
    }
  }
};

struct StatementOwner {
  struct AdbcStatement value{};
  std::shared_ptr<ConnectionOwner> connection;

  ~StatementOwner() {
    if (value.private_data != nullptr) {
      struct AdbcError adbc_error{};
      AdbcStatementRelease(&value, &adbc_error);
      release_error(&adbc_error);
    }
  }
};

struct AdbcDatabaseResource {
  std::shared_ptr<DatabaseOwner> owner = std::make_shared<DatabaseOwner>();
  struct AdbcDatabase &value = owner->value;

  void destructor(ErlNifEnv *env) { let_go(std::move(owner)); }
};
FINE_RESOURCE(AdbcDatabaseResource);

struct AdbcConnectionResource {
  std::shared_ptr<ConnectionOwner> owner = std::make_shared<ConnectionOwner>();
  struct AdbcConnection &value = owner->value;

  void destructor(ErlNifEnv *env) { let_go(std::move(owner)); }
};
FINE_RESOURCE(AdbcConnectionResource);

struct AdbcStatementResource {
  std::shared_ptr<StatementOwner> owner = std::make_shared<StatementOwner>();
  struct AdbcStatement &value = owner->value;

  void destructor(ErlNifEnv *env) { let_go(std::move(owner)); }
};
FINE_RESOURCE(AdbcStatementResource);

struct ArrowArrayStreamResource {
  // Keep the connection alive while reading: a statement's stream through
  // its statement's owner, a metadata stream (get_info, get_objects,
  // get_table_types) through `connection`. Declared first so that they are
  // destroyed last: members are destroyed in reverse order, and the stream
  // must be released while the statement and its connection still exist.
  std::shared_ptr<ConnectionOwner> connection;
  std::shared_ptr<StatementOwner> statement;
  nanoarrow::UniqueArrayStream stream;
  nanoarrow::UniqueSchema schema;
  // Serializes reads against release. The process reading a stream may be
  // killed while it is inside a dirty NIF, and the connection then releases
  // the stream from another scheduler while the read is still running.
  std::mutex mutex;

  // No reader holds the resource once its destructor runs, so the mutex is
  // not needed. The job releases the stream, then the schema, and only then
  // lets go of the statement and the connection. If the job cannot be
  // allocated, all of it is released here, as before.
  void destructor(ErlNifEnv *env) {
    struct Parts {
      std::shared_ptr<ConnectionOwner> connection;
      std::shared_ptr<StatementOwner> statement;
      nanoarrow::UniqueArrayStream stream;
      nanoarrow::UniqueSchema schema;
    };

    std::shared_ptr<Parts> parts;

    try {
      parts = std::make_shared<Parts>();
    } catch (...) {
      stream.reset();
      schema.reset();
      return;
    }

    parts->connection = std::move(connection);
    parts->statement = std::move(statement);
    parts->stream = std::move(stream);
    parts->schema = std::move(schema);

    Releaser::run([parts = std::move(parts)]() mutable {
      parts->stream.reset();
      parts->schema.reset();
      parts.reset();
    });
  }
};
FINE_RESOURCE(ArrowArrayStreamResource);

struct AdbcExecuteOnGCResource {
  ErlNifPid pid{};
  std::string statement;

  void destructor(ErlNifEnv *env) {
    auto msg_env = enif_alloc_env();
    if (msg_env) {
      auto msg =
          fine::encode(msg_env, std::tuple(atoms::execute_on_gc, statement));
      enif_send(NULL, &pid, msg_env, msg);
      enif_free_env(msg_env);
    }
  }
};
FINE_RESOURCE(AdbcExecuteOnGCResource);

// Type alias for ADBC results
template <typename... T>
using AdbcResult = std::variant<fine::Ok<T...>, fine::Error<ExAdbcError>,
                                fine::Error<ExArgumentError>>;

// Get/set option template helpers

template <typename ResType, typename GetString, typename GetBytes,
          typename GetInt, typename GetDouble>
fine::Term adbc_get_option_impl(ErlNifEnv *env, fine::ResourcePtr<ResType> res,
                                const fine::Atom &type, const std::string &key,
                                GetString get_string, GetBytes get_bytes,
                                GetInt get_int, GetDouble get_double) {
  struct AdbcError adbc_error{};

  if (type == "string" || type == "binary") {
    int is_string = (type == "string");
    uint8_t value[64] = {'\0'};
    constexpr size_t value_buffer_size = sizeof(value) / sizeof(value[0]);
    size_t value_len = value_buffer_size;
    AdbcStatusCode code;
    size_t elem_size;

    if (is_string) {
      elem_size = sizeof(char);
      code = get_string(&res->value, key.c_str(), (char *)value, &value_len,
                        &adbc_error);
    } else {
      elem_size = sizeof(uint8_t);
      code =
          get_bytes(&res->value, key.c_str(), value, &value_len, &adbc_error);
    }
    if (code != ADBC_STATUS_OK) {
      return fine::encode(env, fine::Error(ExAdbcError(&adbc_error)));
    }

    if (value_len > value_buffer_size) {
      uint8_t *out_value = (uint8_t *)enif_alloc(elem_size * (value_len + 1));
      memset(out_value, 0, elem_size * (value_len + 1));
      size_t len2 = value_len + 1;
      if (is_string) {
        code = get_string(&res->value, key.c_str(), (char *)out_value, &len2,
                          &adbc_error);
      } else {
        code =
            get_bytes(&res->value, key.c_str(), out_value, &len2, &adbc_error);
      }
      if (code != ADBC_STATUS_OK) {
        enif_free(out_value);
        return fine::encode(env, fine::Error(ExAdbcError(&adbc_error)));
      }
      // minus 1 to remove the null terminator for strings
      auto ret = fine::make_new_binary(env, (const char *)out_value,
                                       value_len - (is_string ? 1 : 0));
      enif_free(out_value);
      return fine::encode(env, fine::Ok(fine::Term(ret)));
    } else {
      // minus 1 to remove the null terminator for strings
      auto ret = fine::make_new_binary(env, (const char *)value,
                                       value_len - (is_string ? 1 : 0));
      return fine::encode(env, fine::Ok(fine::Term(ret)));
    }
  } else if (type == "integer") {
    int64_t value = 0;
    auto code = get_int(&res->value, key.c_str(), &value, &adbc_error);
    if (code != ADBC_STATUS_OK) {
      return fine::encode(env, fine::Error(ExAdbcError(&adbc_error)));
    }
    return fine::encode(env, fine::Ok(fine::Term(enif_make_int64(env, value))));
  } else if (type == "float") {
    double value = 0;
    auto code = get_double(&res->value, key.c_str(), &value, &adbc_error);
    if (code != ADBC_STATUS_OK) {
      return fine::encode(env, fine::Error(ExAdbcError(&adbc_error)));
    }
    return fine::encode(env,
                        fine::Ok(fine::Term(enif_make_double(env, value))));
  } else {
    throw std::invalid_argument(
        "invalid option type, expected :string, :binary, :integer, or :float");
  }
}

template <typename ResType, typename SetString, typename SetBytes,
          typename SetInt, typename SetDouble>
AdbcResult<>
adbc_set_option_impl(ErlNifEnv *env, fine::ResourcePtr<ResType> res,
                     const fine::Atom &type, const std::string &key,
                     fine::Term value_term, SetString set_string,
                     SetBytes set_bytes, SetInt set_int, SetDouble set_double) {
  struct AdbcError adbc_error{};
  AdbcStatusCode code;

  if (type == "string") {
    auto value = fine::decode<std::string>(env, value_term);
    code = set_string(&res->value, key.c_str(), value.c_str(), &adbc_error);
  } else if (type == "binary") {
    auto bin = fine::decode<ErlNifBinary>(env, value_term);
    code = set_bytes(&res->value, key.c_str(), bin.data, bin.size, &adbc_error);
  } else if (type == "integer") {
    auto value = fine::decode<int64_t>(env, value_term);
    code = set_int(&res->value, key.c_str(), value, &adbc_error);
  } else if (type == "float") {
    auto value = fine::decode<double>(env, value_term);
    code = set_double(&res->value, key.c_str(), value, &adbc_error);
  } else {
    throw std::invalid_argument(
        "invalid option type, expected :string, :binary, :integer, or :float");
  }

  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  return fine::Ok<>();
}

// NIF functions

AdbcResult<fine::ResourcePtr<AdbcDatabaseResource>>
adbc_database_new(ErlNifEnv *env) {
  auto db = fine::make_resource<AdbcDatabaseResource>();
  struct AdbcError adbc_error{};
  auto code = AdbcDatabaseNew(&db->value, &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  return fine::Ok(db);
}
FINE_NIF(adbc_database_new, 0);

fine::Term adbc_database_get_option(ErlNifEnv *env,
                                    fine::ResourcePtr<AdbcDatabaseResource> db,
                                    fine::Atom type, std::string key) {
  return adbc_get_option_impl(
      env, db, type, key, AdbcDatabaseGetOption, AdbcDatabaseGetOptionBytes,
      AdbcDatabaseGetOptionInt, AdbcDatabaseGetOptionDouble);
}
FINE_NIF(adbc_database_get_option, 0);

AdbcResult<>
adbc_database_set_option(ErlNifEnv *env,
                         fine::ResourcePtr<AdbcDatabaseResource> db,
                         fine::Atom type, std::string key, fine::Term value) {
  return adbc_set_option_impl(env, db, type, key, value, AdbcDatabaseSetOption,
                              AdbcDatabaseSetOptionBytes,
                              AdbcDatabaseSetOptionInt,
                              AdbcDatabaseSetOptionDouble);
}
FINE_NIF(adbc_database_set_option, 0);

AdbcResult<> adbc_database_init(ErlNifEnv *env,
                                fine::ResourcePtr<AdbcDatabaseResource> db) {
  struct AdbcError adbc_error{};
  auto code = AdbcDatabaseInit(&db->value, &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  return fine::Ok<>();
}
FINE_NIF(adbc_database_init, ERL_NIF_DIRTY_JOB_IO_BOUND);

AdbcResult<fine::ResourcePtr<AdbcConnectionResource>>
adbc_connection_new(ErlNifEnv *env) {
  auto conn = fine::make_resource<AdbcConnectionResource>();
  struct AdbcError adbc_error{};
  auto code = AdbcConnectionNew(&conn->value, &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  return fine::Ok(conn);
}
FINE_NIF(adbc_connection_new, 0);

fine::Term
adbc_connection_get_option(ErlNifEnv *env,
                           fine::ResourcePtr<AdbcConnectionResource> conn,
                           fine::Atom type, std::string key) {
  return adbc_get_option_impl(env, conn, type, key, AdbcConnectionGetOption,
                              AdbcConnectionGetOptionBytes,
                              AdbcConnectionGetOptionInt,
                              AdbcConnectionGetOptionDouble);
}
FINE_NIF(adbc_connection_get_option, 0);

AdbcResult<>
adbc_connection_set_option(ErlNifEnv *env,
                           fine::ResourcePtr<AdbcConnectionResource> conn,
                           fine::Atom type, std::string key, fine::Term value) {
  return adbc_set_option_impl(
      env, conn, type, key, value, AdbcConnectionSetOption,
      AdbcConnectionSetOptionBytes, AdbcConnectionSetOptionInt,
      AdbcConnectionSetOptionDouble);
}
FINE_NIF(adbc_connection_set_option, 0);

AdbcResult<>
adbc_connection_cancel(ErlNifEnv *env,
                       fine::ResourcePtr<AdbcConnectionResource> conn) {
  struct AdbcError adbc_error{};
  auto code = AdbcConnectionCancel(&conn->value, &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  return fine::Ok<>();
}
FINE_NIF(adbc_connection_cancel, ERL_NIF_DIRTY_JOB_IO_BOUND);

AdbcResult<>
adbc_connection_init(ErlNifEnv *env,
                     fine::ResourcePtr<AdbcConnectionResource> conn,
                     fine::ResourcePtr<AdbcDatabaseResource> db) {
  struct AdbcError adbc_error{};
  auto code = AdbcConnectionInit(&conn->value, &db->value, &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  conn->owner->database = db->owner;
  return fine::Ok<>();
}
FINE_NIF(adbc_connection_init, ERL_NIF_DIRTY_JOB_IO_BOUND);

AdbcResult<fine::ResourcePtr<ArrowArrayStreamResource>>
adbc_connection_get_info(ErlNifEnv *env,
                         fine::ResourcePtr<AdbcConnectionResource> conn,
                         std::vector<uint64_t> info_codes) {
  auto stream_res = fine::make_resource<ArrowArrayStreamResource>();
  std::vector<uint32_t> info_codes_u32(info_codes.begin(), info_codes.end());
  uint32_t *ptr = nullptr;
  size_t info_codes_length = info_codes_u32.size();
  if (info_codes_length != 0) {
    ptr = info_codes_u32.data();
  }
  struct AdbcError adbc_error{};
  stream_res->connection = conn->owner;
  auto code = AdbcConnectionGetInfo(&conn->value, ptr, info_codes_length,
                                    stream_res->stream.get(), &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  return fine::Ok(stream_res);
}
FINE_NIF(adbc_connection_get_info, ERL_NIF_DIRTY_JOB_IO_BOUND);

AdbcResult<fine::ResourcePtr<ArrowArrayStreamResource>>
adbc_connection_get_objects(
    ErlNifEnv *env, fine::ResourcePtr<AdbcConnectionResource> conn,
    int64_t depth, std::optional<std::string> catalog,
    std::optional<std::string> db_schema, std::optional<std::string> table_name,
    std::optional<std::vector<std::string>> table_type_opt,
    std::optional<std::string> column_name) {
  const char *catalog_p = catalog ? catalog->c_str() : nullptr;
  const char *db_schema_p = db_schema ? db_schema->c_str() : nullptr;
  const char *table_name_p = table_name ? table_name->c_str() : nullptr;
  const char *column_name_p = column_name ? column_name->c_str() : nullptr;

  std::vector<std::string> table_type_strs;
  std::vector<const char *> table_types;

  if (table_type_opt) {
    table_type_strs = std::move(*table_type_opt);
    for (const auto &tt : table_type_strs) {
      table_types.emplace_back(tt.c_str());
    }
  }
  // Terminate the list with a NULL entry.
  table_types.emplace_back(nullptr);

  auto stream_res = fine::make_resource<ArrowArrayStreamResource>();
  struct AdbcError adbc_error{};
  stream_res->connection = conn->owner;
  auto code = AdbcConnectionGetObjects(
      &conn->value, (int)depth, catalog_p, db_schema_p, table_name_p,
      table_types.data(), column_name_p, stream_res->stream.get(), &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  return fine::Ok(stream_res);
}
FINE_NIF(adbc_connection_get_objects, ERL_NIF_DIRTY_JOB_IO_BOUND);

AdbcResult<fine::ResourcePtr<ArrowArrayStreamResource>>
adbc_connection_get_table_types(
    ErlNifEnv *env, fine::ResourcePtr<AdbcConnectionResource> conn) {
  auto stream_res = fine::make_resource<ArrowArrayStreamResource>();
  struct AdbcError adbc_error{};
  stream_res->connection = conn->owner;
  auto code = AdbcConnectionGetTableTypes(
      &conn->value, stream_res->stream.get(), &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  return fine::Ok(stream_res);
}
FINE_NIF(adbc_connection_get_table_types, ERL_NIF_DIRTY_JOB_IO_BOUND);

AdbcResult<fine::ResourcePtr<AdbcStatementResource>>
adbc_statement_new(ErlNifEnv *env,
                   fine::ResourcePtr<AdbcConnectionResource> conn) {
  auto stmt = fine::make_resource<AdbcStatementResource>();
  struct AdbcError adbc_error{};
  auto code = AdbcStatementNew(&conn->value, &stmt->value, &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  stmt->owner->connection = conn->owner;
  return fine::Ok(stmt);
}
FINE_NIF(adbc_statement_new, 0);

fine::Term
adbc_statement_get_option(ErlNifEnv *env,
                          fine::ResourcePtr<AdbcStatementResource> stmt,
                          fine::Atom type, std::string key) {
  return adbc_get_option_impl(
      env, stmt, type, key, AdbcStatementGetOption, AdbcStatementGetOptionBytes,
      AdbcStatementGetOptionInt, AdbcStatementGetOptionDouble);
}
FINE_NIF(adbc_statement_get_option, 0);

AdbcResult<>
adbc_statement_set_option(ErlNifEnv *env,
                          fine::ResourcePtr<AdbcStatementResource> stmt,
                          fine::Atom type, std::string key, fine::Term value) {
  return adbc_set_option_impl(
      env, stmt, type, key, value, AdbcStatementSetOption,
      AdbcStatementSetOptionBytes, AdbcStatementSetOptionInt,
      AdbcStatementSetOptionDouble);
}
FINE_NIF(adbc_statement_set_option, 0);

AdbcResult<fine::ResourcePtr<ArrowArrayStreamResource>, int64_t>
adbc_statement_execute_query(ErlNifEnv *env,
                             fine::ResourcePtr<AdbcStatementResource> stmt) {
  auto stream_res = fine::make_resource<ArrowArrayStreamResource>();
  int64_t rows_affected = 0;
  struct AdbcError adbc_error{};
  auto code = AdbcStatementExecuteQuery(&stmt->value, stream_res->stream.get(),
                                        &rows_affected, &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  stream_res->statement = stmt->owner;
  return fine::Ok(stream_res, rows_affected);
}
FINE_NIF(adbc_statement_execute_query, ERL_NIF_DIRTY_JOB_IO_BOUND);

AdbcResult<int64_t>
adbc_statement_execute(ErlNifEnv *env,
                       fine::ResourcePtr<AdbcStatementResource> stmt) {
  int64_t rows_affected = 0;
  struct AdbcError adbc_error{};
  auto code = AdbcStatementExecuteQuery(&stmt->value, nullptr, &rows_affected,
                                        &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  return fine::Ok(rows_affected);
}
FINE_NIF(adbc_statement_execute, ERL_NIF_DIRTY_JOB_IO_BOUND);

AdbcResult<>
adbc_statement_prepare(ErlNifEnv *env,
                       fine::ResourcePtr<AdbcStatementResource> stmt) {
  struct AdbcError adbc_error{};
  auto code = AdbcStatementPrepare(&stmt->value, &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  return fine::Ok<>();
}
FINE_NIF(adbc_statement_prepare, ERL_NIF_DIRTY_JOB_IO_BOUND);

AdbcResult<>
adbc_statement_set_sql_query(ErlNifEnv *env,
                             fine::ResourcePtr<AdbcStatementResource> stmt,
                             std::string query) {
  struct AdbcError adbc_error{};
  auto code =
      AdbcStatementSetSqlQuery(&stmt->value, query.c_str(), &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  return fine::Ok<>();
}
FINE_NIF(adbc_statement_set_sql_query, ERL_NIF_DIRTY_JOB_IO_BOUND);

AdbcResult<> adbc_statement_bind(ErlNifEnv *env,
                                 fine::ResourcePtr<AdbcStatementResource> stmt,
                                 std::vector<ExAdbcColumn> columns) {
  nanoarrow::UniqueArray arr;
  nanoarrow::UniqueSchema schema;
  try {
    columns_to_arrow_record_batch(env, columns, arr.get(), schema.get());
  } catch (const nanoarrow::Exception &error) {
    return fine::Error(ExArgumentError(error.what()));
  } catch (const std::invalid_argument &error) {
    return fine::Error(ExArgumentError(error.what()));
  }

  struct AdbcError adbc_error{};
  auto code =
      AdbcStatementBind(&stmt->value, arr.get(), schema.get(), &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  return fine::Ok<>();
}
FINE_NIF(adbc_statement_bind, ERL_NIF_DIRTY_JOB_IO_BOUND);

AdbcResult<>
adbc_statement_bind_stream(ErlNifEnv *env,
                           fine::ResourcePtr<AdbcStatementResource> stmt,
                           fine::ResourcePtr<ArrowArrayStreamResource> stream) {
  struct AdbcError adbc_error{};
  auto code =
      AdbcStatementBindStream(&stmt->value, stream->stream.get(), &adbc_error);
  if (code != ADBC_STATUS_OK) {
    return fine::Error(ExAdbcError(&adbc_error));
  }
  return fine::Ok<>();
}
FINE_NIF(adbc_statement_bind_stream, ERL_NIF_DIRTY_JOB_IO_BOUND);

uint64_t adbc_arrow_array_stream_get_pointer(
    ErlNifEnv *env, fine::ResourcePtr<ArrowArrayStreamResource> res) {
  return reinterpret_cast<uint64_t>(res->stream.get());
}
FINE_NIF(adbc_arrow_array_stream_get_pointer, 0);

fine::Ok<fine::ResourcePtr<ArrowArrayStreamResource>>
adbc_arrow_array_stream_from_pointer(ErlNifEnv *env, uint64_t pointer) {
  // We want to take full ownership of the stream, so we move it into
  // our resource-managed memory. We are the ones responsible for
  // releasing the stream.
  auto source = reinterpret_cast<struct ArrowArrayStream *>(pointer);
  auto stream_res = fine::make_resource<ArrowArrayStreamResource>();
  ArrowArrayStreamMove(source, stream_res->stream.get());
  return fine::Ok(stream_res);
}
FINE_NIF(adbc_arrow_array_stream_from_pointer, 0);

AdbcResult<fine::Term>
adbc_arrow_array_stream_next(ErlNifEnv *env,
                             fine::ResourcePtr<ArrowArrayStreamResource> res) {
  nanoarrow::UniqueArray array;
  fine::Term out_term;
  std::lock_guard<std::mutex> lock(res->mutex);

  if (res->stream->release == nullptr) {
    return fine::Error(ExArgumentError("arrow array stream has been released"));
  }

  if (res->stream->get_next == nullptr) {
    return fine::Error(ExArgumentError("invalid arrow array stream"));
  }

  int code = res->stream->get_next(res->stream.get(), array.get());
  if (code != 0) {
    const char *reason = res->stream->get_last_error(res->stream.get());
    return fine::Error(ExArgumentError(
        reason ? reason : "unknown error: cannot get next record"));
  }

  // if no error and the array is released, the stream has ended
  if (array->release == nullptr) {
    return fine::Ok(fine::encode(env, atoms::end_of_series));
  }

  // only fetch schema once for the entire stream
  if (res->schema->release == nullptr) {
    code = res->stream->get_schema(res->stream.get(), res->schema.get());
    if (code != 0) {
      const char *reason = res->stream->get_last_error(res->stream.get());
      res->schema.reset();
      return fine::Error(ExArgumentError(reason ? reason : "unknown error"));
    }
  }

  try {
    auto columns_term =
        arrow_record_batch_to_columns(env, res->schema.get(), array.get());
    return fine::Ok(columns_term);
  } catch (const nanoarrow::Exception &e) {
    return fine::Error(ExArgumentError(std::string(e.what())));
  } catch (const std::invalid_argument &e) {
    return fine::Error(ExArgumentError(std::string(e.what())));
  }
}
FINE_NIF(adbc_arrow_array_stream_next, ERL_NIF_DIRTY_JOB_IO_BOUND);

fine::Ok<> adbc_arrow_array_stream_release(
    ErlNifEnv *env, fine::ResourcePtr<ArrowArrayStreamResource> res) {
  std::lock_guard<std::mutex> lock(res->mutex);
  res->stream.reset();
  return fine::Ok<>();
}
FINE_NIF(adbc_arrow_array_stream_release, ERL_NIF_DIRTY_JOB_IO_BOUND);

fine::Term adbc_ipc_system_endianness(ErlNifEnv *env) {
  if (ArrowIpcSystemEndianness() == NANOARROW_IPC_ENDIANNESS_BIG) {
    return fine::encode(env, atoms::big);
  } else {
    return fine::encode(env, atoms::little);
  }
}
FINE_NIF(adbc_ipc_system_endianness, 0);

AdbcResult<fine::ResourcePtr<ArrowArrayStreamResource>>
adbc_ipc_load_stream_binary(ErlNifEnv *env, ErlNifBinary binary) {
  nanoarrow::UniqueBuffer input_buffer;

  auto code = ArrowBufferAppend(input_buffer.get(), binary.data, binary.size);
  if (code != NANOARROW_OK) {
    return fine::Error(ExArgumentError(
        "Failed to append binary data to Arrow IPC input buffer"));
  }

  struct ArrowIpcInputStream input;
  code = ArrowIpcInputStreamInitBuffer(&input, input_buffer.get());
  if (code != NANOARROW_OK) {
    return fine::Error(
        ExArgumentError("Failed to initialize Arrow IPC array stream"));
  }

  auto stream_res = fine::make_resource<ArrowArrayStreamResource>();
  code =
      ArrowIpcArrayStreamReaderInit(stream_res->stream.get(), &input, nullptr);
  if (code != NANOARROW_OK) {
    return fine::Error(
        ExArgumentError("Failed to initialize Arrow IPC array stream reader"));
  }

  return fine::Ok(stream_res);
}
FINE_NIF(adbc_ipc_load_stream_binary, ERL_NIF_DIRTY_JOB_CPU_BOUND);

AdbcResult<std::string>
adbc_ipc_dump_stream_ref(ErlNifEnv *env,
                         fine::ResourcePtr<ArrowArrayStreamResource> res) {
  std::lock_guard<std::mutex> lock(res->mutex);

  if (res->stream->release == nullptr) {
    return fine::Error(ExArgumentError("stream has already been consumed"));
  }

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream ostream;
  int code = ArrowIpcOutputStreamInitBuffer(ostream.get(), output.get());
  if (code != NANOARROW_OK) {
    return fine::Error(ExArgumentError("invalid Arrow IPC output stream"));
  }

  nanoarrow::ipc::UniqueWriter writer;
  code = ArrowIpcWriterInit(writer.get(), ostream.get());
  if (code != NANOARROW_OK) {
    return fine::Error(ExArgumentError("invalid Arrow IPC writer"));
  }

  struct ArrowError arrow_error{};
  code = ArrowIpcWriterWriteArrayStream(writer.get(), res->stream.get(),
                                        &arrow_error);

  res->stream.reset();

  if (code != NANOARROW_OK) {
    return fine::Error(ExArgumentError(arrow_error.message));
  }

  auto binary = std::string(reinterpret_cast<const char *>(output->data),
                            output->size_bytes);

  return fine::Ok(binary);
}
FINE_NIF(adbc_ipc_dump_stream_ref, ERL_NIF_DIRTY_JOB_CPU_BOUND);

fine::ResourcePtr<AdbcExecuteOnGCResource>
adbc_execute_on_gc_new(ErlNifEnv *env, ErlNifPid pid, std::string statement) {
  auto res = fine::make_resource<AdbcExecuteOnGCResource>();
  res->pid = pid;
  res->statement = std::move(statement);
  return res;
}
FINE_NIF(adbc_execute_on_gc_new, 0);

void drain_releases_on_halt(void *) { Releaser::drain(Releaser::halt_timeout); }

// fine's load callback, plus the on-halt callback that lets queued releases
// finish before the VM exits. The VM only takes that callback at load.
int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  if (fine::__private__::load(env, priv_data, load_info) != 0) {
    return -1;
  }

  if (enif_set_option(env, ERL_NIF_OPT_ON_HALT, drain_releases_on_halt) != 0) {
    return -1;
  }

  return 0;
}

} // namespace adbc_nif

// FINE_INIT, with adbc_nif::load as the load callback.
ERL_NIF_INIT_PROLOGUE
ERL_NIF_INIT_GLOB
ERL_NIF_INIT_DECL(NAME);
ERL_NIF_INIT_DECL(NAME) {
  auto &nif_funcs = fine::__private__::get_erl_nif_funcs();
  static ErlNifEntry entry = {ERL_NIF_MAJOR_VERSION,
                              ERL_NIF_MINOR_VERSION,
                              "Elixir.Adbc.Nif",
                              static_cast<int>(nif_funcs.size()),
                              nif_funcs.data(),
                              adbc_nif::load,
                              NULL,
                              NULL,
                              NULL,
                              ERL_NIF_VM_VARIANT,
                              1,
                              sizeof(ErlNifResourceTypeInit),
                              ERL_NIF_MIN_ERTS_VERSION};
  ERL_NIF_INIT_BODY;
  return &entry;
}
ERL_NIF_INIT_EPILOGUE
