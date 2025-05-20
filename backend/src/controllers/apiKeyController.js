/**
 * API Key Controller
 * Handles HTTP requests related to API key management
 */

const apiKeyService = require('../services/apiKeyService');

/**
 * Adds a new API key for a user
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
async function addApiKey(req, res) {
  try {
    const { apiKey, apiSecret, scopes } = req.body;
    const userId = req.user.id; // Assuming authentication middleware sets req.user

    // Validate inputs
    if (!apiKey || !apiSecret) {
      return res.status(400).json({ 
        success: false, 
        message: 'API key and secret are required' 
      });
    }

    // Validate API key format
    if (!apiKeyService.validateApiKeyFormat(apiKey)) {
      return res.status(400).json({ 
        success: false, 
        message: 'Invalid API key format' 
      });
    }

    // Validate API secret format
    if (!apiKeyService.validateApiSecretFormat(apiSecret)) {
      return res.status(400).json({ 
        success: false, 
        message: 'Invalid API secret format' 
      });
    }

    // Add the API key
    const result = await apiKeyService.addApiKey(userId, apiKey, apiSecret, scopes);

    return res.status(201).json({
      success: true,
      message: 'API key added successfully',
      data: result
    });
  } catch (error) {
    console.error('Error adding API key:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'Failed to add API key', 
      error: error.message 
    });
  }
}

/**
 * Retrieves a user's API keys (masked)
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
async function getApiKeys(req, res) {
  try {
    const userId = req.user.id; // Assuming authentication middleware sets req.user
    
    const apiKeys = await apiKeyService.listApiKeys(userId);
    
    return res.status(200).json({
      success: true,
      data: apiKeys
    });
  } catch (error) {
    console.error('Error retrieving API keys:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'Failed to retrieve API keys', 
      error: error.message 
    });
  }
}

/**
 * Revokes (deletes) an API key
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
async function revokeApiKey(req, res) {
  try {
    const { keyId } = req.params;
    const userId = req.user.id; // Assuming authentication middleware sets req.user
    
    if (!keyId) {
      return res.status(400).json({ 
        success: false, 
        message: 'API key ID is required' 
      });
    }
    
    const result = await apiKeyService.revokeApiKey(keyId, userId);
    
    if (!result) {
      return res.status(404).json({ 
        success: false, 
        message: 'API key not found or already revoked' 
      });
    }
    
    return res.status(200).json({
      success: true,
      message: 'API key revoked successfully'
    });
  } catch (error) {
    console.error('Error revoking API key:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'Failed to revoke API key', 
      error: error.message 
    });
  }
}

/**
 * Validates an API key (for testing purposes)
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
async function validateApiKey(req, res) {
  try {
    const userId = req.user.id; // Assuming authentication middleware sets req.user
    
    // Retrieve the API key
    const apiKeyData = await apiKeyService.getApiKey(userId);
    
    if (!apiKeyData) {
      return res.status(404).json({ 
        success: false, 
        message: 'No valid API key found' 
      });
    }
    
    // For security, we don't return the actual key/secret
    return res.status(200).json({
      success: true,
      message: 'API key is valid and was successfully decrypted',
      data: {
        id: apiKeyData.id,
        userId: apiKeyData.userId,
        scopes: apiKeyData.scopes,
        expiry: apiKeyData.expiry,
        createdAt: apiKeyData.createdAt
      }
    });
  } catch (error) {
    console.error('Error validating API key:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'Failed to validate API key', 
      error: error.message 
    });
  }
}

module.exports = {
  addApiKey,
  getApiKeys,
  revokeApiKey,
  validateApiKey
}; 