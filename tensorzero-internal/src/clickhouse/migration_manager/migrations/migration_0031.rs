use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

use super::{check_table_exists, get_column_type};
use async_trait::async_trait;

/// This migration extends the BatchRequest status enum to support OpenAI-compatible batch statuses.
///
/// The original enum only supported: 'pending', 'completed', 'failed'
/// The new enum will support all OpenAI batch statuses: 'pending', 'completed', 'failed',
/// 'validating', 'in_progress', 'finalizing', 'expired', 'cancelling', 'cancelled'
pub struct Migration0031<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0031";

#[async_trait]
impl Migration for Migration0031<'_> {
    /// Check that the BatchRequest table exists
    async fn can_apply(&self) -> Result<(), Error> {
        if !check_table_exists(self.clickhouse, "BatchRequest", MIGRATION_ID).await? {
            return Err(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "BatchRequest table does not exist".to_string(),
            }
            .into());
        }
        Ok(())
    }

    /// Check if the migration has already been applied by checking the enum values
    async fn should_apply(&self) -> Result<bool, Error> {
        // First check if we're in a transitional state
        let batch_request_new_exists = check_table_exists(self.clickhouse, "BatchRequest_new", MIGRATION_ID).await?;
        if batch_request_new_exists {
            // We're in the middle of migration, should continue applying
            return Ok(true);
        }

        // Check if BatchRequest table exists
        let batch_request_exists = check_table_exists(self.clickhouse, "BatchRequest", MIGRATION_ID).await?;
        if !batch_request_exists {
            // Table doesn't exist, can't apply migration
            return Ok(false);
        }

        // Check if the status column exists
        use super::check_column_exists;
        if !check_column_exists(self.clickhouse, "BatchRequest", "status", MIGRATION_ID).await? {
            // No status column, something is wrong
            return Ok(false);
        }

        let column_type =
            get_column_type(self.clickhouse, "BatchRequest", "status", MIGRATION_ID).await?;

        // Check if the enum already includes the new OpenAI statuses
        if column_type.contains("validating")
            && column_type.contains("in_progress")
            && column_type.contains("finalizing")
            && column_type.contains("expired")
            && column_type.contains("cancelling")
            && column_type.contains("cancelled")
        {
            return Ok(false);
        }

        Ok(true)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // ClickHouse doesn't support direct enum modification, so we need to:
        // 1. Create a new table with the extended enum
        // 2. Copy data from the old table to the new table
        // 3. Drop the old table
        // 4. Rename the new table to the original name
        //
        // This approach avoids ALTER TABLE UPDATE which is not supported in older ClickHouse versions

        // Check if we're in a transitional state
        let batch_request_new_exists = check_table_exists(self.clickhouse, "BatchRequest_new", MIGRATION_ID).await?;
        let batch_request_exists = check_table_exists(self.clickhouse, "BatchRequest", MIGRATION_ID).await?;

        if !batch_request_new_exists {
            // Step 1: Create new table with extended enum
            // First, get the current table structure to preserve all columns
            let create_new_table_query = r#"
                CREATE TABLE IF NOT EXISTS BatchRequest_new
                (
                    batch_id UUID,
                    id UUID,
                    batch_params String,
                    model_name LowCardinality(String),
                    model_provider_name LowCardinality(String),
                    status Enum(
                        'pending' = 1, 
                        'completed' = 2, 
                        'failed' = 3,
                        'validating' = 4,
                        'in_progress' = 5,
                        'finalizing' = 6,
                        'expired' = 7,
                        'cancelling' = 8,
                        'cancelled' = 9
                    ) DEFAULT 'pending',
                    errors Map(UUID, String),
                    timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id),
                    openai_batch_id Nullable(String),
                    completion_window String DEFAULT '24h',
                    input_file_id Nullable(String),
                    output_file_id Nullable(String),
                    error_file_id Nullable(String),
                    created_at Nullable(DateTime),
                    in_progress_at Nullable(DateTime),
                    expires_at Nullable(DateTime),
                    finalizing_at Nullable(DateTime),
                    completed_at Nullable(DateTime),
                    failed_at Nullable(DateTime),
                    expired_at Nullable(DateTime),
                    cancelling_at Nullable(DateTime),
                    cancelled_at Nullable(DateTime),
                    request_counts Nullable(String),
                    metadata Nullable(String)
                ) ENGINE = MergeTree()
                ORDER BY (batch_id, id)"#;

            self.clickhouse
                .run_query_synchronous(create_new_table_query.to_string(), None)
                .await?;
        }

        // Step 2: Copy data from old table to new table (if old table still exists)
        if batch_request_exists && batch_request_new_exists {
            let copy_data_query = r#"
                INSERT INTO BatchRequest_new 
                SELECT 
                    batch_id,
                    id,
                    batch_params,
                    model_name,
                    model_provider_name,
                    CAST(status AS Enum(
                        'pending' = 1, 
                        'completed' = 2, 
                        'failed' = 3,
                        'validating' = 4,
                        'in_progress' = 5,
                        'finalizing' = 6,
                        'expired' = 7,
                        'cancelling' = 8,
                        'cancelled' = 9
                    )) as status,
                    errors,
                    openai_batch_id,
                    completion_window,
                    input_file_id,
                    output_file_id,
                    error_file_id,
                    created_at,
                    in_progress_at,
                    expires_at,
                    finalizing_at,
                    completed_at,
                    failed_at,
                    expired_at,
                    cancelling_at,
                    cancelled_at,
                    request_counts,
                    metadata
                FROM BatchRequest"#;

            let copy_result = self.clickhouse
                .run_query_synchronous(copy_data_query.to_string(), None)
                .await;

            match copy_result {
                Ok(_) => {
                    // Successfully copied data
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    // If the table doesn't exist, another process might have already handled it
                    if !error_msg.contains("UNKNOWN_TABLE") && !error_msg.contains("doesn't exist") {
                        return Err(e);
                    }
                }
            }
        }

        // Step 3: Drop the old table
        if batch_request_exists {
            let drop_old_table_query = "DROP TABLE IF EXISTS BatchRequest";
            let drop_result = self.clickhouse
                .run_query_synchronous(drop_old_table_query.to_string(), None)
                .await;

            match drop_result {
                Ok(_) => {},
                Err(e) => {
                    let error_msg = e.to_string();
                    // If the table doesn't exist, that's fine
                    if !error_msg.contains("UNKNOWN_TABLE") && !error_msg.contains("doesn't exist") {
                        return Err(e);
                    }
                }
            }
        }

        // Step 4: Rename new table to original name
        // Only rename if BatchRequest_new exists and BatchRequest doesn't
        let batch_request_new_still_exists = check_table_exists(self.clickhouse, "BatchRequest_new", MIGRATION_ID).await?;
        let batch_request_still_exists = check_table_exists(self.clickhouse, "BatchRequest", MIGRATION_ID).await?;
        
        if batch_request_new_still_exists && !batch_request_still_exists {
            let rename_table_query = "RENAME TABLE BatchRequest_new TO BatchRequest";
            let rename_result = self.clickhouse
                .run_query_synchronous(rename_table_query.to_string(), None)
                .await;

            match rename_result {
                Ok(_) => {},
                Err(e) => {
                    let error_msg = e.to_string();
                    // If the error is because the table doesn't exist or already exists,
                    // another process completed the migration
                    if !error_msg.contains("UNKNOWN_TABLE") 
                        && !error_msg.contains("doesn't exist")
                        && !error_msg.contains("already exists") {
                        return Err(e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        // The migration is successful if:
        // 1. BatchRequest table exists with the new enum values
        // 2. We're in a transitional state (BatchRequest_new exists)

        let batch_request_exists = check_table_exists(self.clickhouse, "BatchRequest", MIGRATION_ID).await?;
        let batch_request_new_exists = check_table_exists(self.clickhouse, "BatchRequest_new", MIGRATION_ID).await?;

        if batch_request_new_exists {
            // We're in a transitional state - consider it succeeded since another process will complete it
            return Ok(true);
        }

        if batch_request_exists {
            // Check if the status column has the new enum values
            let column_type =
                get_column_type(self.clickhouse, "BatchRequest", "status", MIGRATION_ID).await?;

            // If it contains any of the new values, the migration succeeded
            if column_type.contains("validating") || column_type.contains("in_progress") {
                return Ok(true);
            }
        }

        // If neither condition is met, check using the original logic
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }

    fn rollback_instructions(&self) -> String {
        "/* Create a table with the original enum */
CREATE TABLE BatchRequest_old (
    batch_id UUID,
    id UUID,
    batch_params String,
    model_name LowCardinality(String),
    model_provider_name LowCardinality(String),
    status Enum('pending' = 1, 'completed' = 2, 'failed' = 3) DEFAULT 'pending',
    errors Map(UUID, String),
    timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id),
    openai_batch_id Nullable(String),
    completion_window String DEFAULT '24h',
    input_file_id Nullable(String),
    output_file_id Nullable(String),
    error_file_id Nullable(String),
    created_at Nullable(DateTime),
    in_progress_at Nullable(DateTime),
    expires_at Nullable(DateTime),
    finalizing_at Nullable(DateTime),
    completed_at Nullable(DateTime),
    failed_at Nullable(DateTime),
    expired_at Nullable(DateTime),
    cancelling_at Nullable(DateTime),
    cancelled_at Nullable(DateTime),
    request_counts Nullable(String),
    metadata Nullable(String)
) ENGINE = MergeTree()
ORDER BY (batch_id, id);

/* Copy data with status mapping */
INSERT INTO BatchRequest_old 
SELECT 
    batch_id,
    id,
    batch_params,
    model_name,
    model_provider_name,
    CASE 
        WHEN status IN ('validating', 'in_progress', 'finalizing') THEN 'pending' 
        WHEN status = 'completed' THEN 'completed' 
        WHEN status IN ('failed', 'expired', 'cancelled') THEN 'failed' 
        ELSE CAST(status AS Enum('pending' = 1, 'completed' = 2, 'failed' = 3))
    END as status,
    errors,
    openai_batch_id,
    completion_window,
    input_file_id,
    output_file_id,
    error_file_id,
    created_at,
    in_progress_at,
    expires_at,
    finalizing_at,
    completed_at,
    failed_at,
    expired_at,
    cancelling_at,
    cancelled_at,
    request_counts,
    metadata
FROM BatchRequest;

/* Drop and rename */
DROP TABLE BatchRequest;
RENAME TABLE BatchRequest_old TO BatchRequest;"
            .to_string()
    }
}
