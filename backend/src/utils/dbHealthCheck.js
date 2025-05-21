const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

/**
 * Performs a health check on the database connection
 * @returns {Promise<{isHealthy: boolean, message: string, error?: any}>}
 */
async function checkDatabaseConnection() {
  try {
    // Try to run a simple query against the database
    await prisma.$queryRaw`SELECT 1`;
    return {
      isHealthy: true,
      message: 'Database connection is healthy'
    };
  } catch (error) {
    return {
      isHealthy: false,
      message: 'Database connection failed',
      error
    };
  }
}

/**
 * Verifies that all expected tables exist in the database
 * @returns {Promise<{isHealthy: boolean, message: string, missingTables?: string[], error?: any}>}
 */
async function checkSchemaState() {
  try {
    // Get a list of all tables that should exist based on our Prisma schema
    const expectedTables = ['User', 'ApiKey', 'TradeLog', 'Metric'];
    
    // Get actual tables from the database
    const tables = await prisma.$queryRaw`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public'
    `;
    
    // Extract table names from the result
    const existingTables = tables.map(t => t.table_name.toLowerCase());
    
    // Check if all expected tables exist
    const missingTables = expectedTables.filter(
      table => !existingTables.includes(table.toLowerCase())
    );
    
    if (missingTables.length > 0) {
      return {
        isHealthy: false,
        message: 'Database schema is incomplete',
        missingTables
      };
    }
    
    return {
      isHealthy: true,
      message: 'Database schema is complete'
    };
  } catch (error) {
    return {
      isHealthy: false,
      message: 'Schema check failed',
      error
    };
  }
}

/**
 * Performs a comprehensive health check on the database
 * @returns {Promise<{isHealthy: boolean, connection: Object, schema: Object}>}
 */
async function performHealthCheck() {
  const connection = await checkDatabaseConnection();
  
  // Only check schema if connection is healthy
  const schema = connection.isHealthy ? 
    await checkSchemaState() : 
    { isHealthy: false, message: 'Skipped due to connection failure' };
  
  return {
    isHealthy: connection.isHealthy && schema.isHealthy,
    connection,
    schema
  };
}

module.exports = {
  checkDatabaseConnection,
  checkSchemaState,
  performHealthCheck
}; 