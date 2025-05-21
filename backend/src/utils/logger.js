/**
 * Logger Utility
 * Provides consistent logging across the application
 */

// For production, we might want to use a more robust logging solution
// like Winston or Pino, but this simple logger is sufficient for now

/**
 * Logger class with standard log levels
 */
class Logger {
  /**
   * Creates a new logger instance
   * @param {string} module - The module name for context in logs
   */
  constructor(module) {
    this.module = module;
    this.env = process.env.NODE_ENV || 'development';
  }

  /**
   * Formats a log message with timestamp and module
   * @private
   * @param {string} level - Log level
   * @param {string} message - Log message
   * @returns {string} Formatted log message
   */
  _format(level, message) {
    const timestamp = new Date().toISOString();
    return `[${timestamp}] [${level.toUpperCase()}] [${this.module}] ${message}`;
  }

  /**
   * Log an informational message
   * @param {string} message - The message to log
   * @param {Object} [data] - Optional data to include
   */
  info(message, data = null) {
    console.info(this._format('info', message), data ? JSON.stringify(data) : '');
  }

  /**
   * Log a warning message
   * @param {string} message - The message to log
   * @param {Object} [data] - Optional data to include
   */
  warn(message, data = null) {
    console.warn(this._format('warn', message), data ? JSON.stringify(data) : '');
  }

  /**
   * Log an error message
   * @param {string} message - The message to log
   * @param {Error|Object} [error] - Optional error object or data
   */
  error(message, error = null) {
    console.error(this._format('error', message), error instanceof Error 
      ? error.stack 
      : (error ? JSON.stringify(error) : ''));
  }

  /**
   * Log a debug message (only in development)
   * @param {string} message - The message to log
   * @param {Object} [data] - Optional data to include
   */
  debug(message, data = null) {
    if (this.env === 'development') {
      console.debug(this._format('debug', message), data ? JSON.stringify(data) : '');
    }
  }

  /**
   * Redact sensitive information from objects for logging
   * @param {Object} obj - Object to redact
   * @param {Array<string>} fields - Fields to redact
   * @returns {Object} Redacted object
   */
  static redact(obj, fields = ['password', 'apiKey', 'apiSecret', 'token', 'secret', 'signature']) {
    if (!obj || typeof obj !== 'object') return obj;
    
    const redacted = { ...obj };
    
    fields.forEach(field => {
      if (field in redacted) {
        redacted[field] = '***REDACTED***';
      }
    });
    
    return redacted;
  }
}

/**
 * Create a logger for a specific module
 * @param {string} module - Module name
 * @returns {Logger} Logger instance
 */
function createLogger(module) {
  return new Logger(module);
}

module.exports = {
  createLogger,
  Logger
}; 