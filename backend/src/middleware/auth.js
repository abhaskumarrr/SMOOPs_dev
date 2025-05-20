/**
 * Authentication Middleware
 * Placeholder implementation - will be replaced with proper authentication in a future task
 */

/**
 * Middleware to authenticate API requests
 * Currently a placeholder that sets a mock user for development
 *
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
function authenticate(req, res, next) {
  // This is a placeholder implementation
  // In a real implementation, this would verify a JWT token or session
  
  console.log('Auth middleware: Setting mock user for development');
  
  // Set a mock user for development
  req.user = {
    id: '12345678-1234-1234-1234-123456789012',
    name: 'Test User',
    email: 'test@example.com'
  };
  
  // Continue to the next middleware or route handler
  next();
}

module.exports = {
  authenticate
}; 