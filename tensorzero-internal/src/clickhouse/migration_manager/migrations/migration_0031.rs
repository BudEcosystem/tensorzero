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
        // 1. Add a new column with the extended enum
        // 2. Copy data from old column to new column
        // 3. Drop the old column
        // 4. Rename the new column to the original name

        // Step 1: Add new status column with extended enum
        let query = r#"
            ALTER TABLE BatchRequest
            ADD COLUMN IF NOT EXISTS status_new Enum(
                'pending' = 1, 
                'completed' = 2, 
                'failed' = 3,
                'validating' = 4,
                'in_progress' = 5,
                'finalizing' = 6,
                'expired' = 7,
                'cancelling' = 8,
                'cancelled' = 9
            ) DEFAULT 'pending'"#;

        self.clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        // Step 2: Copy data from old status column to new one
        let query = r#"
            ALTER TABLE BatchRequest
            UPDATE status_new = status
            WHERE 1 = 1"#;

        self.clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        // Step 3: Drop the old status column
        let query = r#"
            ALTER TABLE BatchRequest
            DROP COLUMN status"#;

        self.clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        // Step 4: Rename the new column to the original name
        let query = r#"
            ALTER TABLE BatchRequest
            RENAME COLUMN status_new TO status"#;

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
        "/* Revert the status enum back to original values */\
            ALTER TABLE BatchRequest ADD COLUMN status_old Enum('pending' = 1, 'completed' = 2, 'failed' = 3) DEFAULT 'pending';\
            ALTER TABLE BatchRequest UPDATE status_old = CASE \
                WHEN status IN ('validating', 'in_progress', 'finalizing') THEN 'pending' \
                WHEN status = 'completed' THEN 'completed' \
                WHEN status IN ('failed', 'expired', 'cancelled') THEN 'failed' \
                ELSE status \
            END WHERE 1 = 1;\
            ALTER TABLE BatchRequest DROP COLUMN status;\
            ALTER TABLE BatchRequest RENAME COLUMN status_old TO status;"
            .to_string()
    }
}
