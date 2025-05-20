/**
 * API Key Management Service
 * Handles encryption, storage, and retrieval of Delta Exchange API keys
 */

const { PrismaClient } = require('@prisma/client');
const encryption = require('../utils/encryption');
const secureKey = require('../utils/secureKey');

const prisma = new PrismaClient();

/**
 * Adds a new API key for a user
 * @param {string} userId - The ID of the user
 * @param {string} apiKey - The API key to store
 * @param {string} apiSecret - The API secret to store
 * @param {string[]} [scopes=['read', 'trade']] - The permissions/scopes for this key
 * @param {Date} [expiry=null] - Optional expiry date for the key
 * @returns {Promise<Object>} The created API key record (without sensitive data)
 */
async function addApiKey(userId, apiKey, apiSecret, scopes = ['read', 'trade'], expiry = null) {
  if (!userId || !apiKey || !apiSecret) {
    throw new Error('User ID, API key, and secret are required');
  }

  // Validate if user exists
  const user = await prisma.user.findUnique({
    where: { id: userId }
  });

  if (!user) {
    throw new Error(`User with ID ${userId} not found`);
  }

  // Get the master encryption key
  const masterKey = secureKey.getMasterKey();
  
  // Derive a key specific for this user's API keys
  const derivedKey = secureKey.deriveKey(masterKey, 'api-keys', userId);

  // Encrypt the API key and secret
  const encryptedKey = encryption.encrypt(apiKey, derivedKey);
  const encryptedSecret = encryption.encrypt(apiSecret, derivedKey);

  // Format scopes as a comma-separated string
  const scopesStr = Array.isArray(scopes) ? scopes.join(',') : scopes;

  // Store the encrypted data in the database
  const apiKeyRecord = await prisma.apiKey.create({
    data: {
      userId,
      // Store a masked version of the key as a unique identifier
      key: `${maskApiKey(apiKey)}_${encryptedKey.iv}`,
      // Store the encrypted data in JSON format
      encryptedData: JSON.stringify({
        key: encryptedKey,
        secret: encryptedSecret
      }),
      scopes: scopesStr,
      expiry: expiry || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000) // Default 30 days
    }
  });

  return {
    id: apiKeyRecord.id,
    userId: apiKeyRecord.userId,
    maskedKey: maskApiKey(apiKey),
    scopes: apiKeyRecord.scopes.split(','),
    expiry: apiKeyRecord.expiry,
    createdAt: apiKeyRecord.createdAt
  };
}

/**
 * Retrieves a user's API key and secret
 * @param {string} userId - The ID of the user
 * @returns {Promise<Object|null>} The decrypted API key and secret or null if not found
 */
async function getApiKey(userId) {
  if (!userId) {
    throw new Error('User ID is required');
  }

  // Find the user's API key in the database
  const apiKeyRecord = await prisma.apiKey.findFirst({
    where: { 
      userId,
      expiry: { gt: new Date() } // Only retrieve non-expired keys
    },
    orderBy: { createdAt: 'desc' } // Get the most recent key
  });

  if (!apiKeyRecord) {
    return null;
  }

  try {
    // Get the master encryption key
    const masterKey = secureKey.getMasterKey();
    
    // Derive the key specific to this user's API keys
    const derivedKey = secureKey.deriveKey(masterKey, 'api-keys', userId);

    // Parse the encrypted data
    const encryptedData = JSON.parse(apiKeyRecord.encryptedData);
    
    // Decrypt the API key and secret
    const apiKey = encryption.decrypt(encryptedData.key, derivedKey);
    const apiSecret = encryption.decrypt(encryptedData.secret, derivedKey);

    return {
      id: apiKeyRecord.id,
      userId: apiKeyRecord.userId,
      key: apiKey,
      secret: apiSecret,
      scopes: apiKeyRecord.scopes.split(','),
      expiry: apiKeyRecord.expiry,
      createdAt: apiKeyRecord.createdAt
    };
  } catch (error) {
    console.error(`Error decrypting API key for user ${userId}:`, error);
    throw new Error('Failed to decrypt API key data');
  }
}

/**
 * Returns a list of user's API keys with masked values
 * @param {string} userId - The ID of the user
 * @returns {Promise<Array>} Array of API key records with masked sensitive data
 */
async function listApiKeys(userId) {
  if (!userId) {
    throw new Error('User ID is required');
  }

  const apiKeys = await prisma.apiKey.findMany({
    where: { userId },
    orderBy: { createdAt: 'desc' }
  });

  return apiKeys.map(key => ({
    id: key.id,
    userId: key.userId,
    maskedKey: key.key.split('_')[0], // The masked version is stored before the underscore
    scopes: key.scopes.split(','),
    expiry: key.expiry,
    createdAt: key.createdAt,
    isExpired: key.expiry < new Date()
  }));
}

/**
 * Revokes (deletes) an API key
 * @param {string} keyId - The ID of the API key to revoke
 * @param {string} userId - The ID of the user (for verification)
 * @returns {Promise<boolean>} True if successful, false otherwise
 */
async function revokeApiKey(keyId, userId) {
  if (!keyId || !userId) {
    throw new Error('API key ID and user ID are required');
  }

  try {
    await prisma.apiKey.deleteMany({
      where: {
        id: keyId,
        userId
      }
    });
    
    return true;
  } catch (error) {
    console.error(`Error revoking API key ${keyId}:`, error);
    return false;
  }
}

/**
 * Masks an API key for display purposes
 * @param {string} apiKey - The API key to mask
 * @returns {string} Masked version of the key
 */
function maskApiKey(apiKey) {
  if (!apiKey || apiKey.length < 8) {
    return '****';
  }
  
  // Show first 4 and last 4 characters
  return `${apiKey.substring(0, 4)}****${apiKey.substring(apiKey.length - 4)}`;
}

/**
 * Validates the format of a Delta Exchange API key
 * @param {string} apiKey - The API key to validate
 * @returns {boolean} True if valid format, false otherwise
 */
function validateApiKeyFormat(apiKey) {
  // Basic validation: Check length and character set
  // Adjust this based on Delta Exchange API key format
  return typeof apiKey === 'string' && 
         apiKey.length >= 16 && 
         /^[a-zA-Z0-9]+$/.test(apiKey);
}

/**
 * Validates the format of a Delta Exchange API secret
 * @param {string} apiSecret - The API secret to validate
 * @returns {boolean} True if valid format, false otherwise
 */
function validateApiSecretFormat(apiSecret) {
  // Basic validation: Check length and character set
  // Adjust this based on Delta Exchange API secret format
  return typeof apiSecret === 'string' && 
         apiSecret.length >= 32 && 
         /^[a-zA-Z0-9]+$/.test(apiSecret);
}

module.exports = {
  addApiKey,
  getApiKey,
  listApiKeys,
  revokeApiKey,
  validateApiKeyFormat,
  validateApiSecretFormat
}; 