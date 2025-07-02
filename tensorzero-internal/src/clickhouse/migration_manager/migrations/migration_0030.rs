use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

use super::{check_column_exists, check_table_exists};
use async_trait::async_trait;

/// This migration adds OpenAI-compatible batch API fields to the BatchRequest table.
///
/// These fields support OpenAI's batch API format while maintaining backward compatibility
/// with TensorZero's existing batch inference system.
///
/// New fields:
/// - openai_batch_id: OpenAI's batch identifier
/// - completion_window: Time window for batch completion (e.g., "24h")
/// - input_file_id: Reference to uploaded input file
/// - output_file_id: Reference to generated output file
/// - error_file_id: Reference to generated error file
/// - Various timestamp fields for OpenAI batch lifecycle tracking
/// - request_counts: JSON with batch statistics
/// - metadata: JSON with custom metadata
pub struct Migration0030<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0030";

#[async_trait]
impl Migration for Migration0030<'_> {
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

    /// Check if the migration has already been applied by checking if the OpenAI-specific columns exist
    async fn should_apply(&self) -> Result<bool, Error> {
        // Since all columns are added in a single atomic ALTER TABLE statement,
        // we only need to check for one of them to see if the migration has been applied.
        Ok(!check_column_exists(
            self.clickhouse,
            "BatchRequest",
            "openai_batch_id",
            MIGRATION_ID,
        )
        .await?)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // Add OpenAI-specific columns to the BatchRequest table
        let query = r#"
            ALTER TABLE BatchRequest
            ADD COLUMN IF NOT EXISTS openai_batch_id Nullable(String),
            ADD COLUMN IF NOT EXISTS completion_window String DEFAULT '24h',
            ADD COLUMN IF NOT EXISTS input_file_id Nullable(String),
            ADD COLUMN IF NOT EXISTS output_file_id Nullable(String),
            ADD COLUMN IF NOT EXISTS error_file_id Nullable(String),
            ADD COLUMN IF NOT EXISTS created_at Nullable(DateTime),
            ADD COLUMN IF NOT EXISTS in_progress_at Nullable(DateTime),
            ADD COLUMN IF NOT EXISTS expires_at Nullable(DateTime),
            ADD COLUMN IF NOT EXISTS finalizing_at Nullable(DateTime),
            ADD COLUMN IF NOT EXISTS completed_at Nullable(DateTime),
            ADD COLUMN IF NOT EXISTS failed_at Nullable(DateTime),
            ADD COLUMN IF NOT EXISTS expired_at Nullable(DateTime),
            ADD COLUMN IF NOT EXISTS cancelling_at Nullable(DateTime),
            ADD COLUMN IF NOT EXISTS cancelled_at Nullable(DateTime),
            ADD COLUMN IF NOT EXISTS request_counts Nullable(String),
            ADD COLUMN IF NOT EXISTS metadata Nullable(String);"#;

        self.clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        Ok(())
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }

    fn rollback_instructions(&self) -> String {
        "/* Drop the OpenAI batch API columns */\
            ALTER TABLE BatchRequest \
            DROP COLUMN openai_batch_id,\
            DROP COLUMN completion_window,\
            DROP COLUMN input_file_id,\
            DROP COLUMN output_file_id,\
            DROP COLUMN error_file_id,\
            DROP COLUMN created_at,\
            DROP COLUMN in_progress_at,\
            DROP COLUMN expires_at,\
            DROP COLUMN finalizing_at,\
            DROP COLUMN completed_at,\
            DROP COLUMN failed_at,\
            DROP COLUMN expired_at,\
            DROP COLUMN cancelling_at,\
            DROP COLUMN cancelled_at,\
            DROP COLUMN request_counts,\
            DROP COLUMN metadata;"
            .to_string()
    }
}
