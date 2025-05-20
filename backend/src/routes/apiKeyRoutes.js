/**
 * API Key Routes
 * Defines routes for API key management
 */

const express = require('express');
const apiKeyController = require('../controllers/apiKeyController');
const { authenticate } = require('../middleware/auth'); // This will be implemented later

const router = express.Router();

// Apply authentication middleware to all routes
router.use(authenticate);

// Route to add a new API key
router.post('/', apiKeyController.addApiKey);

// Route to get all API keys (masked) for the authenticated user
router.get('/', apiKeyController.getApiKeys);

// Route to revoke (delete) an API key
router.delete('/:keyId', apiKeyController.revokeApiKey);

// Route to validate an API key (for testing purposes)
router.get('/validate', apiKeyController.validateApiKey);

module.exports = router; 