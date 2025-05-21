/**
 * Environment configuration utility for SMOOPs backend
 * Handles environment variable validation and provides defaults
 */

require('dotenv').config({ path: process.cwd() + '/../.env' });
const path = require('path');
const fs = require('fs');

// Check for root-level .env file
if (!fs.existsSync(path.join(process.cwd(), '..', '.env'))) {
  console.warn('\x1b[33m%s\x1b[0m', 'WARNING: No .env file found at project root. Using default values.');
}

// Environment variables with defaults
const env = {
  // Node environment (development, production, test)
  NODE_ENV: process.env.NODE_ENV || 'development',
  
  // Server configuration
  PORT: parseInt(process.env.PORT || '3001', 10),
  HOST: process.env.HOST || '0.0.0.0',
  
  // Database configuration
  DATABASE_URL: process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/smoops?schema=public',
  
  // JWT configuration for authentication
  JWT_SECRET: process.env.JWT_SECRET || 'dev-jwt-secret-do-not-use-in-production',
  JWT_EXPIRES_IN: process.env.JWT_EXPIRES_IN || '7d',
  
  // CORS configuration
  CORS_ORIGIN: process.env.CORS_ORIGIN || '*',
  
  // Encryption for API keys
  ENCRYPTION_MASTER_KEY: process.env.ENCRYPTION_MASTER_KEY || 'development_key_do_not_use_in_production',
  
  // Exchange configuration
  DELTA_EXCHANGE_TESTNET: process.env.DELTA_EXCHANGE_TESTNET === 'false' ? false : true,
  DELTA_EXCHANGE_API_URL: process.env.DELTA_EXCHANGE_API_URL || 'https://testnet-api.delta.exchange',
  
  // ML service configuration
  ML_SERVICE_URL: process.env.ML_SERVICE_URL || 'http://localhost:3002',
  
  // Logging configuration
  LOG_LEVEL: process.env.LOG_LEVEL || 'info',
};

// Validate critical environment variables
function validateEnvironment() {
  const errors = [];
  
  // Check for production with default secret
  if (env.NODE_ENV === 'production') {
    if (env.JWT_SECRET === 'dev-jwt-secret-do-not-use-in-production') {
      errors.push('JWT_SECRET is using default value in production mode');
    }
    
    if (env.ENCRYPTION_MASTER_KEY === 'development_key_do_not_use_in_production') {
      errors.push('ENCRYPTION_MASTER_KEY is using default value in production mode');
    }
    
    if (env.CORS_ORIGIN === '*') {
      errors.push('CORS_ORIGIN should not be * in production mode');
    }
  }
  
  // Validate DATABASE_URL format
  const dbUrlPattern = /^postgresql:\/\/.+:.+@.+:\d+\/.+(\?.*)?$/;
  if (!dbUrlPattern.test(env.DATABASE_URL)) {
    errors.push('DATABASE_URL is invalid or missing');
  }
  
  // Log errors and exit if in production
  if (errors.length > 0) {
    console.error('\x1b[31m%s\x1b[0m', 'Environment validation errors:');
    errors.forEach(error => console.error(`- ${error}`));
    
    if (env.NODE_ENV === 'production') {
      console.error('\x1b[31m%s\x1b[0m', 'Exiting due to environment validation errors in production mode.');
      process.exit(1);
    }
  }
}

// In development, log the environment configuration
if (env.NODE_ENV === 'development') {
  console.log('\x1b[36m%s\x1b[0m', 'Environment Configuration:');
  // Strip sensitive values
  const logSafeEnv = { ...env };
  // Mask sensitive values
  logSafeEnv.JWT_SECRET = logSafeEnv.JWT_SECRET ? '********' : 'not set';
  logSafeEnv.ENCRYPTION_MASTER_KEY = logSafeEnv.ENCRYPTION_MASTER_KEY ? '********' : 'not set';
  logSafeEnv.DATABASE_URL = logSafeEnv.DATABASE_URL.replace(/\/\/(.+):(.+)@/, '//******:******@');
  
  // Log safe environment
  console.log(JSON.stringify(logSafeEnv, null, 2));
}

// Run validation
validateEnvironment();

module.exports = env; 