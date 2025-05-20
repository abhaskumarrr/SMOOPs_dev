/**
 * Encryption utility for API keys using AES-256-GCM
 * Securely encrypts and decrypts sensitive data like Delta Exchange API keys
 */

const crypto = require('crypto');

// Algorithm details
const ALGORITHM = 'aes-256-gcm';
const IV_LENGTH = 12; // Recommended for GCM
const KEY_LENGTH = 32; // 256 bits
const AUTH_TAG_LENGTH = 16; // 128 bits

/**
 * Generates a cryptographically secure encryption key
 * @returns {Buffer} 32-byte encryption key
 */
const generateEncryptionKey = () => {
  return crypto.randomBytes(KEY_LENGTH);
};

/**
 * Encrypts sensitive data using AES-256-GCM
 * @param {string} text - Plain text to encrypt
 * @param {Buffer|string} encryptionKey - Key used for encryption
 * @returns {Object} - Object containing encrypted content, iv, and authTag
 */
const encrypt = (text, encryptionKey) => {
  if (!text || !encryptionKey) {
    throw new Error('Text and encryption key are required');
  }

  // Convert key to Buffer if provided as string
  const key = typeof encryptionKey === 'string' 
    ? Buffer.from(encryptionKey, 'hex') 
    : encryptionKey;

  // Generate a random initialization vector for each encryption
  const iv = crypto.randomBytes(IV_LENGTH);
  
  // Create cipher with the key and iv
  const cipher = crypto.createCipheriv(ALGORITHM, key, iv);
  
  // Encrypt the text
  let encrypted = cipher.update(text, 'utf8', 'hex');
  encrypted += cipher.final('hex');
  
  // Get the authentication tag
  const authTag = cipher.getAuthTag().toString('hex');
  
  return {
    encryptedData: encrypted,
    iv: iv.toString('hex'),
    authTag
  };
};

/**
 * Decrypts data that was encrypted using encrypt function
 * @param {Object} encryptedData - Object containing encrypted content, iv, and authTag
 * @param {Buffer|string} encryptionKey - Key used for encryption
 * @returns {string} - Decrypted text
 */
const decrypt = ({ encryptedData, iv, authTag }, encryptionKey) => {
  if (!encryptedData || !iv || !authTag || !encryptionKey) {
    throw new Error('Encrypted data, IV, authTag, and encryption key are required');
  }

  // Convert key to Buffer if provided as string
  const key = typeof encryptionKey === 'string' 
    ? Buffer.from(encryptionKey, 'hex') 
    : encryptionKey;

  // Convert IV and authTag to Buffer if they're strings
  const ivBuffer = typeof iv === 'string' ? Buffer.from(iv, 'hex') : iv;
  const authTagBuffer = typeof authTag === 'string' ? Buffer.from(authTag, 'hex') : authTag;

  // Create decipher
  const decipher = crypto.createDecipheriv(ALGORITHM, key, ivBuffer);
  
  // Set auth tag
  decipher.setAuthTag(authTagBuffer);
  
  // Decrypt
  let decrypted = decipher.update(encryptedData, 'hex', 'utf8');
  decrypted += decipher.final('utf8');
  
  return decrypted;
};

module.exports = {
  generateEncryptionKey,
  encrypt,
  decrypt,
  KEY_LENGTH,
  IV_LENGTH,
  AUTH_TAG_LENGTH
}; 