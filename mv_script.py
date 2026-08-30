# Databricks notebook source

from pathlib import Path
from pyspark import pipelines as dp


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Location of the SQL files.
#
# Every MV has exactly one SQL file:
#
#     <mv_name>.sql
#
# Example:
#
#     charger_location_mv.sql
#
# The SQL file contains ONLY the SELECT statement.
# -----------------------------------------------------------------------------

SQL_FOLDER = "/Workspace/Repos/<repo>/<project>/sql"


# -----------------------------------------------------------------------------
# Environment-specific catalog
#
# The catalog is NOT hardcoded.
#
# Example:
#     DEV  -> catalog A
#     TEST -> catalog B
#     PROD -> catalog C
#
# The secret scope/key below should be replaced with your company's
# Secrets Manager / Databricks secret configuration.
# -----------------------------------------------------------------------------

SECRET_SCOPE = "<secret-scope>"
CATALOG_SECRET_KEY = "<environment-catalog-secret-key>"


# Get catalog from the environment-specific secret.
CATALOG = dbutils.secrets.get(
    scope=SECRET_SCOPE,
    key=CATALOG_SECRET_KEY
).strip()


# -----------------------------------------------------------------------------
# Materialized View configuration
#
# IMPORTANT:
# Only MV metadata is maintained here.
#
# The SQL filename is automatically derived:
#
#     <name>.sql
#
# So:
#
#     name = "charger_location_mv"
#
# automatically maps to:
#
#     charger_location_mv.sql
# -----------------------------------------------------------------------------

MVS = [

    # =====================================================================
    # CHAIN 1
    # mapping → location → evse → connector
    #                              ├── session
    #                              └── information → information_cxm
    # =====================================================================

    {
        "name": "mapping_charger_id_connector_mv",
        "schema": "euh-emobility",
        "comment": (
            "Maps charger IDs to connector IDs."
        ),
    },

    {
        "name": "charger_location_mv",
        "schema": "euh-emobility",
        "comment": (
            "Provides charger location information."
        ),
    },

    {
        "name": "charger_evse_mv",
        "schema": "euh-emobility",
        "comment": (
            "Provides charger EVSE information."
        ),
    },

    {
        "name": "charger_connector_mv",
        "schema": "euh-emobility",
        "comment": (
            "Provides charger connector information."
        ),
    },

    {
        "name": "charger_session_mv",
        "schema": "euh-emobility",
        "comment": (
            "Provides charger session information."
        ),
    },

    {
        "name": "charger_information_mv",
        "schema": "curated-emob-datahub-reporting",
        "comment": (
            "Provides charger information for the reporting layer."
        ),
    },

    {
        "name": "charger_information_cxm",
        "schema": "curated-emob-datahub-srs",
        "comment": (
            "Provides CXM charger information."
        ),
    },

    # =====================================================================
    # CHAIN 2
    # charger_detail → charger_summary
    # =====================================================================

    {
        "name": "charger_detail_mv",
        "schema": "curated-emob-datahub-reporting",
        "comment": (
            "Provides detailed charger information for reporting."
        ),
    },

    {
        "name": "charger_summary_mv",
        "schema": "curated-emob-datahub-reporting",
        "comment": (
            "Provides aggregated charger summary information."
        ),
    },
]


# =============================================================================
# 2. VALIDATION
# =============================================================================

if not CATALOG:
    raise ValueError(
        "Catalog could not be retrieved from the configured secret."
    )


if not MVS:
    raise ValueError(
        "No materialized views have been configured."
    )


# Prevent accidental duplicate MV names.
mv_names = [mv["name"] for mv in MVS]

if len(mv_names) != len(set(mv_names)):
    raise ValueError(
        "Duplicate materialized view name detected in MVS configuration."
    )


# =============================================================================
# 3. READ SQL FILE
# =============================================================================

def read_mv_sql(mv_name: str) -> str:
    """
    Reads the SELECT statement for the materialized view.

    File convention:
        <mv_name>.sql

    Example:
        charger_location_mv
            ->
        charger_location_mv.sql
    """

    sql_path = Path(SQL_FOLDER) / f"{mv_name}.sql"

    if not sql_path.exists():
        raise FileNotFoundError(
            f"SQL file not found for MV '{mv_name}': {sql_path}"
        )

    sql = open(
        sql_path,
        "r",
        encoding="utf-8"
    ).read().strip()

    if not sql:
        raise ValueError(
            f"SQL file is empty for MV '{mv_name}': {sql_path}"
        )

    return sql


# =============================================================================
# 4. REGISTER MATERIALIZED VIEWS
# =============================================================================
#
# Each iteration dynamically creates a pipeline dataset definition equivalent
# to:
#
#     CREATE OR REFRESH MATERIALIZED VIEW
#     <catalog>.<schema>.<mv_name>
#     REFRESH POLICY AUTO
#     COMMENT ...
#     AS
#     <contents of <mv_name>.sql>
#
# The SQL file itself contains ONLY the SELECT statement.
#
# Lakeflow evaluates all registered datasets together and derives the DAG from
# the dependencies contained in the SELECT statements.
#
# =============================================================================

for mv in MVS:

    mv_name = mv["name"]
    mv_schema = mv["schema"]
    mv_comment = mv["comment"]

    # Fully qualified target name.
    #
    # Example:
    #
    # `my_catalog`.`euh-emobility`.`charger_location_mv`
    #
    target_name = (
        f"`{CATALOG}`."
        f"`{mv_schema}`."
        f"`{mv_name}`"
    )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Capture the current MV values as default arguments.
    # This prevents Python's late-binding behavior when functions are created
    # inside the loop.
    # -------------------------------------------------------------------------

    def create_mv(
        name=mv_name,
        sql_schema=mv_schema,
        comment=mv_comment,
        target=target_name,
    ):

        @dp.materialized_view(
            name=target,
            comment=comment,
            refresh_policy="auto"
        )
        def materialized_view_definition():

            # SQL file is read when the pipeline evaluates this dataset.
            query = read_mv_sql(name)

            # Execute the SELECT and return the DataFrame that defines the MV.
            return spark.sql(query)

        return materialized_view_definition

    # -------------------------------------------------------------------------
    # Register the generated function as a module-level pipeline definition.
    #
    # The loop itself does NOT execute the MVs sequentially.
    #
    # It only creates the dataset definitions.
    #
    # Lakeflow subsequently analyzes all definitions and constructs the DAG
    # based on the references inside the SQL queries.
    # -------------------------------------------------------------------------

    registered_definition = create_mv()

    globals()[f"define_{mv_name}"] = registered_definition
