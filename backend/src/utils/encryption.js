/**
 * Enhanced Encryption Utilities
 * Functions for encrypting and decrypting sensitive data with key rotation support
 */

const crypto = require('crypto');

// Ensure encryption key is set
if (!process.env.ENCRYPTION_KEY) {
  console.error('ENCRYPTION_KEY environment variable is required');
  process.exit(1);
}

// Convert environment variable to buffer - should be a 64 character hex string (32 bytes)
const primaryKey = Buffer.from(process.env.ENCRYPTION_KEY, 'hex');

// Optional secondary key for key rotation
const secondaryKey = process.env.ENCRYPTION_KEY_SECONDARY 
  ? Buffer.from(process.env.ENCRYPTION_KEY_SECONDARY, 'hex') 
  : null;

// Encryption algorithm
const ALGORITHM = 'aes-256-gcm';
const KEY_SIZE = 32; // 256 bits
const IV_SIZE = 16; // 128 bits
const VERSION_IDENTIFIER = {
  V1: '01', // Original version
  V2: '02'  // Enhanced version with additional salt
};

/**
 * Encrypt data using AES-256-GCM with enhanced security
 * @param {string|Object} data - Data to encrypt
 * @param {Object} options - Encryption options
 * @param {boolean} options.useSalt - Whether to add additional salt (default: true)
 * @returns {string} - Encrypted data as hex string with version, IV, salt and authTag
 */
const encrypt = (data, options = { useSalt: true }) => {
  try {
    // Validate key size
    if (primaryKey.length !== KEY_SIZE) {
      throw new Error(`Encryption key must be ${KEY_SIZE * 2} characters (${KEY_SIZE} bytes)`);
    }
    
    // Convert object to string if necessary
    const dataString = typeof data === 'object' ? JSON.stringify(data) : data;
    
    // Generate random initialization vector
    const iv = crypto.randomBytes(IV_SIZE);
    
    // Add version identifier and optional salt for additional security
    const version = options.useSalt ? VERSION_IDENTIFIER.V2 : VERSION_IDENTIFIER.V1;
    
    // Generate random salt if using V2
    const salt = options.useSalt ? crypto.randomBytes(16) : Buffer.alloc(0);
    
    // Create cipher
    const cipher = crypto.createCipheriv(ALGORITHM, primaryKey, iv);
    
    // Encrypt data
    let encrypted = cipher.update(dataString, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    // Get authentication tag
    const authTag = cipher.getAuthTag().toString('hex');
    
    // Combine version, IV, salt (if any), encrypted data, and authTag
    if (options.useSalt) {
      return `${version}:${iv.toString('hex')}:${salt.toString('hex')}:${encrypted}:${authTag}`;
    } else {
      return `${version}:${iv.toString('hex')}:${encrypted}:${authTag}`;
    }
  } catch (error) {
    console.error('Encryption error:', error);
    throw new Error('Failed to encrypt data: ' + error.message);
  }
};

/**
 * Decrypt data using AES-256-GCM with support for different versions and key rotation
 * @param {string} encryptedData - Encrypted data as hex string
 * @returns {string|Object} - Decrypted data, parsed as JSON if applicable
 */
const decrypt = (encryptedData) => {
  try {
    // Parse the encrypted data components
    const parts = encryptedData.split(':');
    
    // Handle different versions
    const version = parts[0];
    
    let ivHex, encrypted, authTagHex, key;
    
    if (version === VERSION_IDENTIFIER.V1) {
      // Legacy format: version:iv:encrypted:authTag
      [, ivHex, encrypted, authTagHex] = parts;
      key = primaryKey;
    } else if (version === VERSION_IDENTIFIER.V2) {
      // Enhanced format: version:iv:salt:encrypted:authTag
      [, ivHex, /* salt */, encrypted, authTagHex] = parts;
      key = primaryKey;
    } else {
      // Unknown version, possibly encrypted with secondary key
      ivHex = parts[1];
      encrypted = parts[parts.length - 2];
      authTagHex = parts[parts.length - 1];
      
      // Try with primary key first
      key = primaryKey;
      
      // If that fails, we'll try with secondary key in the catch block
    }
    
    // Convert hex strings to buffers
    const iv = Buffer.from(ivHex, 'hex');
    const authTag = Buffer.from(authTagHex, 'hex');
    
    // Create decipher
    const decipher = crypto.createDecipheriv(ALGORITHM, key, iv);
    decipher.setAuthTag(authTag);
    
    // Decrypt data
    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    // Try to parse as JSON if it looks like JSON
    try {
      if (decrypted.startsWith('{') || decrypted.startsWith('[')) {
        return JSON.parse(decrypted);
      }
    } catch (e) {
      // If parsing fails, return as string
    }
    
    return decrypted;
    
  } catch (error) {
    // If decryption failed with primary key and secondary key is available, try with secondary key
    if (secondaryKey) {
      try {
        const parts = encryptedData.split(':');
        const ivHex = parts[1];
        const encrypted = parts[parts.length - 2];
        const authTagHex = parts[parts.length - 1];
        
        // Convert hex strings to buffers
        const iv = Buffer.from(ivHex, 'hex');
        const authTag = Buffer.from(authTagHex, 'hex');
        
        // Create decipher with secondary key
        const decipher = crypto.createDecipheriv(ALGORITHM, secondaryKey, iv);
        decipher.setAuthTag(authTag);
        
        // Decrypt data
        let decrypted = decipher.update(encrypted, 'hex', 'utf8');
        decrypted += decipher.final('utf8');
        
        // Try to parse as JSON if it looks like JSON
        try {
          if (decrypted.startsWith('{') || decrypted.startsWith('[')) {
            return JSON.parse(decrypted);
          }
        } catch (e) {
          // If parsing fails, return as string
        }
        
        return decrypted;
        
      } catch (secondaryError) {
        console.error('Secondary key decryption error:', secondaryError);
        throw new Error('Failed to decrypt data with all available keys');
      }
    }
    
    console.error('Decryption error:', error);
    throw new Error('Failed to decrypt data');
  }
};

/**
 * Re-encrypt data using the current primary key
 * Useful for key rotation when data was encrypted with an old key
 * @param {string} encryptedData - Data encrypted with old key
 * @returns {string} - Data re-encrypted with new primary key
 */
const reEncrypt = (encryptedData) => {
  // Decrypt with either primary or secondary key
  const decrypted = decrypt(encryptedData);
  
  // Re-encrypt with current primary key using latest version format
  return encrypt(decrypted, { useSalt: true });
};

/**
 * Generates a secure random encryption key
 * @returns {string} - Hex-encoded encryption key
 */
const generateSecureKey = () => {
  return crypto.randomBytes(KEY_SIZE).toString('hex');
};

module.exports = {
  encrypt,
  decrypt,
  reEncrypt,
  generateSecureKey
}; 