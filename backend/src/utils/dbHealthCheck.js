/**
 * Database Health Check Utility
 * Functions for checking database connectivity and health
 */

const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

/**
 * Checks the database connection health
 * @returns {Promise<Object>} The health check result
 */
async function checkDbConnection() {
  const startTime = Date.now();
  
  try {
    // Execute a simple query to verify database connection
    await prisma.$queryRaw`SELECT 1`;
    
    // Calculate query execution time
    const responseTime = Date.now() - startTime;
    
    return {
      success: true,
      message: 'Database connection is healthy',
      responseTime: `${responseTime}ms`
    };
  } catch (error) {
    return {
      success: false,
      message: 'Database connection failed',
      error: error.message,
      responseTime: `${Date.now() - startTime}ms`
    };
  }
}

/**
 * Performs a comprehensive database health check
 * @returns {Promise<Object>} Detailed health check result
 */
async function performDatabaseHealthCheck() {
  try {
    // Check connection
    const connectionStatus = await checkDbConnection();
    
    // If connection failed, return early
    if (!connectionStatus.success) {
      return {
        success: false,
        connection: connectionStatus,
        tables: null,
        migrations: null
      };
    }
    
    // Check tables (users)
    let userTableStatus = { success: true };
    try {
      await prisma.user.findFirst();
    } catch (error) {
      userTableStatus = {
        success: false,
        error: error.message
      };
    }
    
    // Check migrations table
    let migrationsStatus = { success: true };
    try {
      await prisma.$queryRaw`SELECT * FROM "_prisma_migrations" LIMIT 1`;
    } catch (error) {
      migrationsStatus = {
        success: false,
        error: error.message
      };
    }
    
    return {
      success: connectionStatus.success && userTableStatus.success && migrationsStatus.success,
      connection: connectionStatus,
      tables: {
        users: userTableStatus
      },
      migrations: migrationsStatus
    };
  } catch (error) {
    return {
      success: false,
      message: 'Comprehensive health check failed',
      error: error.message
    };
  }
}

module.exports = {
  checkDbConnection,
  performDatabaseHealthCheck
}; 