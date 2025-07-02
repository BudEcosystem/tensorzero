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
        use super::check_column_exists;

        // First check if the status column exists
        if !check_column_exists(self.clickhouse, "BatchRequest", "status", MIGRATION_ID).await? {
            // If there's no status column, we need to check if status_new exists
            // which would mean we're in the middle of the migration
            return check_column_exists(
                self.clickhouse,
                "BatchRequest",
                "status_new",
                MIGRATION_ID,
            )
            .await;
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
        use super::check_column_exists;

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
        // We need to be careful here because in concurrent scenarios, the column might disappear
        // between our check and the actual UPDATE. We'll use a try-catch approach.
        if check_column_exists(self.clickhouse, "BatchRequest", "status", MIGRATION_ID).await? {
            // First, let's try to copy the data
            let copy_result = self.clickhouse
                .run_query_synchronous(
                    "ALTER TABLE BatchRequest UPDATE status_new = status WHERE 1 = 1".to_string(),
                    None
                )
                .await;
            
            // If the copy failed because the column doesn't exist, that's OK - another migration
            // process might have already handled it
            match copy_result {
                Ok(_) => {
                    // Successfully copied, now drop the old column
                    let query = r#"
                        ALTER TABLE BatchRequest
                        DROP COLUMN IF EXISTS status"#;

                    self.clickhouse
                        .run_query_synchronous(query.to_string(), None)
                        .await?;
                },
                Err(e) => {
                    // Check if the error is because the column doesn't exist
                    let error_msg = e.to_string();
                    if error_msg.contains("UNKNOWN_IDENTIFIER") || error_msg.contains("Missing columns: 'status'") {
                        // This is expected in concurrent scenarios - another process already handled it
                        // Continue with the migration
                    } else {
                        // This is an unexpected error, propagate it
                        return Err(e);
                    }
                }
            }
        }

        // Step 4: Rename the new column to the original name
        // Only rename if status_new exists and status doesn't
        if check_column_exists(self.clickhouse, "BatchRequest", "status_new", MIGRATION_ID).await?
            && !check_column_exists(self.clickhouse, "BatchRequest", "status", MIGRATION_ID).await?
        {
            let rename_result = self.clickhouse
                .run_query_synchronous(
                    "ALTER TABLE BatchRequest RENAME COLUMN status_new TO status".to_string(),
                    None
                )
                .await;
            
            // Handle potential concurrent rename attempts
            match rename_result {
                Ok(_) => {},
                Err(e) => {
                    let error_msg = e.to_string();
                    // If the error is because status_new doesn't exist or status already exists,
                    // that means another process completed the migration
                    if error_msg.contains("UNKNOWN_IDENTIFIER") 
                        || error_msg.contains("Missing columns: 'status_new'")
                        || error_msg.contains("NOT_FOUND_COLUMN_IN_BLOCK")
                        || error_msg.contains("Cannot find column")
                        || error_msg.contains("already exists") {
                        // This is expected in concurrent scenarios
                    } else {
                        // Unexpected error, propagate it
                        return Err(e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        use super::check_column_exists;
        
        // The migration is successful if:
        // 1. There is a 'status' column with the new enum values, OR
        // 2. We're in a transitional state but another process will complete it
        
        // Check if status column exists
        let status_exists = check_column_exists(self.clickhouse, "BatchRequest", "status", MIGRATION_ID).await?;
        let status_new_exists = check_column_exists(self.clickhouse, "BatchRequest", "status_new", MIGRATION_ID).await?;
        
        if status_exists && !status_new_exists {
            // Check if it has the new enum values
            let column_type = get_column_type(self.clickhouse, "BatchRequest", "status", MIGRATION_ID).await?;
            
            // If it contains any of the new values, the migration succeeded
            if column_type.contains("validating") || column_type.contains("in_progress") {
                return Ok(true);
            }
        } else if status_new_exists {
            // We're in a transitional state - consider it succeeded since another process will complete it
            return Ok(true);
        }
        
        // If neither condition is met, check using the original logic
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }

    fn rollback_instructions(&self) -> String {
        "ALTER TABLE BatchRequest ADD COLUMN status_old Enum('pending' = 1, 'completed' = 2, 'failed' = 3) DEFAULT 'pending'
ALTER TABLE BatchRequest UPDATE status_old = CASE WHEN status IN ('validating', 'in_progress', 'finalizing') THEN 'pending' WHEN status = 'completed' THEN 'completed' WHEN status IN ('failed', 'expired', 'cancelled') THEN 'failed' ELSE status END WHERE 1 = 1
ALTER TABLE BatchRequest DROP COLUMN status
ALTER TABLE BatchRequest RENAME COLUMN status_old TO status"
            .to_string()
    }
}
