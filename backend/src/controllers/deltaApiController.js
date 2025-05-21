/**
 * Delta Exchange API Controller
 * Handles Delta Exchange API interactions
 */

const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();
const { decrypt } = require('../utils/encryption');
const axios = require('axios');
const crypto = require('crypto');

// Delta API service class
class DeltaApiService {
  constructor(apiKey, apiSecret, isTestnet = false) {
    this.apiKey = apiKey;
    this.apiSecret = apiSecret;
    this.baseUrl = isTestnet 
      ? 'https://testnet.delta.exchange/api/v2' 
      : 'https://api.delta.exchange/v2';
  }

  /**
   * Sign a request with HMAC signature
   * @param {string} method - HTTP method
   * @param {string} path - Endpoint path
   * @param {Object} params - Query params or body data
   * @returns {Object} - Headers with signature
   */
  createSignedHeaders(method, path, params = {}) {
    const timestamp = Math.floor(Date.now() / 1000);
    let message = timestamp + method + path;
    
    // Add query params or body params if present
    if (Object.keys(params).length > 0) {
      const sortedParams = Object.keys(params).sort().reduce((acc, key) => {
        acc[key] = params[key];
        return acc;
      }, {});
      const paramString = new URLSearchParams(sortedParams).toString();
      message += paramString;
    }
    
    // Create HMAC signature
    const signature = crypto
      .createHmac('sha256', this.apiSecret)
      .update(message)
      .digest('hex');
    
    return {
      'api-key': this.apiKey,
      'timestamp': timestamp.toString(),
      'signature': signature
    };
  }

  /**
   * Execute a request to Delta Exchange API
   * @param {string} method - HTTP method
   * @param {string} endpoint - API endpoint
   * @param {Object} params - Query params or body data
   * @returns {Promise<Object>} - API response
   */
  async request(method, endpoint, params = {}) {
    try {
      const url = `${this.baseUrl}${endpoint}`;
      const headers = this.createSignedHeaders(method, endpoint, params);
      const config = { 
        method, 
        url, 
        headers
      };
      
      // Add params as query or body depending on method
      if (method === 'GET' && Object.keys(params).length > 0) {
        config.params = params;
      } else if (Object.keys(params).length > 0) {
        config.data = params;
      }
      
      const response = await axios(config);
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(
          JSON.stringify({
            status: error.response.status,
            message: error.response.data.message || 'API error',
            data: error.response.data
          })
        );
      }
      throw error;
    }
  }

  // Market data methods
  async getProducts() {
    return this.request('GET', '/products');
  }

  async getOrderBook(productId) {
    return this.request('GET', `/orderbooks/${productId}`);
  }

  async getRecentTrades(productId, limit = 100) {
    return this.request('GET', `/trades`, { 
      product_id: productId, 
      limit 
    });
  }

  // Account methods
  async getWalletBalance() {
    return this.request('GET', '/wallet/balances');
  }

  async getPositions() {
    return this.request('GET', '/positions');
  }

  // Order methods
  async getOrders(status = 'live') {
    return this.request('GET', '/orders', { status });
  }

  async createOrder(orderData) {
    return this.request('POST', '/orders', orderData);
  }

  async cancelOrder(orderId) {
    return this.request('DELETE', `/orders/${orderId}`);
  }
}

/**
 * Helper function to get Delta API service for authenticated user
 * @param {Object} user - User object from request
 * @returns {Promise<DeltaApiService>} - DeltaApiService instance
 */
async function getDeltaApiService(user) {
  // Get the most recent API key for user
  const apiKeyRecord = await prisma.apiKey.findFirst({
    where: { userId: user.id },
    orderBy: { createdAt: 'desc' }
  });

  if (!apiKeyRecord) {
    throw new Error('No API key found for user');
  }

  // Decrypt API key data
  const apiKeyData = decrypt(apiKeyRecord.encryptedData);
  
  // Return Delta API service instance
  return new DeltaApiService(
    apiKeyData.apiKey,
    apiKeyData.apiSecret,
    apiKeyData.testnet
  );
}

/**
 * Get available products
 * @route GET /api/delta/products
 * @access Private
 */
const getProducts = async (req, res) => {
  try {
    const deltaApi = await getDeltaApiService(req.user);
    const data = await deltaApi.getProducts();
    
    res.status(200).json({
      success: true,
      data
    });
  } catch (error) {
    console.error('Get products error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching products',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Get order book for a product
 * @route GET /api/delta/products/:id/orderbook
 * @access Private
 */
const getOrderBook = async (req, res) => {
  try {
    const { id } = req.params;
    const deltaApi = await getDeltaApiService(req.user);
    const data = await deltaApi.getOrderBook(id);
    
    res.status(200).json({
      success: true,
      data
    });
  } catch (error) {
    console.error('Get order book error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching order book',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Get recent trades for a product
 * @route GET /api/delta/products/:id/trades
 * @access Private
 */
const getRecentTrades = async (req, res) => {
  try {
    const { id } = req.params;
    const { limit } = req.query;
    const deltaApi = await getDeltaApiService(req.user);
    const data = await deltaApi.getRecentTrades(id, limit);
    
    res.status(200).json({
      success: true,
      data
    });
  } catch (error) {
    console.error('Get recent trades error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching recent trades',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Get account balance
 * @route GET /api/delta/balance
 * @access Private
 */
const getAccountBalance = async (req, res) => {
  try {
    const deltaApi = await getDeltaApiService(req.user);
    const data = await deltaApi.getWalletBalance();
    
    res.status(200).json({
      success: true,
      data
    });
  } catch (error) {
    console.error('Get account balance error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching account balance',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Get positions
 * @route GET /api/delta/positions
 * @access Private
 */
const getPositions = async (req, res) => {
  try {
    const deltaApi = await getDeltaApiService(req.user);
    const data = await deltaApi.getPositions();
    
    res.status(200).json({
      success: true,
      data
    });
  } catch (error) {
    console.error('Get positions error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching positions',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Get orders
 * @route GET /api/delta/orders
 * @access Private
 */
const getOrders = async (req, res) => {
  try {
    const { status } = req.query;
    const deltaApi = await getDeltaApiService(req.user);
    const data = await deltaApi.getOrders(status || 'live');
    
    res.status(200).json({
      success: true,
      data
    });
  } catch (error) {
    console.error('Get orders error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching orders',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Create order
 * @route POST /api/delta/orders
 * @access Private
 */
const createOrder = async (req, res) => {
  try {
    const orderData = req.body;
    
    // Validate order data
    if (!orderData.product_id || !orderData.size || !orderData.side || !orderData.order_type) {
      return res.status(400).json({
        success: false,
        message: 'Missing required order parameters'
      });
    }
    
    const deltaApi = await getDeltaApiService(req.user);
    const data = await deltaApi.createOrder(orderData);
    
    // Log trade to database
    try {
      await prisma.tradeLog.create({
        data: {
          userId: req.user.id,
          instrument: orderData.product_id.toString(),
          amount: parseFloat(orderData.size),
          price: parseFloat(orderData.limit_price || '0'),
          timestamp: new Date()
        }
      });
    } catch (logError) {
      console.error('Error logging trade:', logError);
    }
    
    res.status(201).json({
      success: true,
      data
    });
  } catch (error) {
    console.error('Create order error:', error);
    
    // Try to parse error from Delta API
    let errorMessage = 'Error creating order';
    let errorData = {};
    
    try {
      const parsedError = JSON.parse(error.message);
      errorMessage = parsedError.message;
      errorData = parsedError.data;
    } catch (e) {
      // Use original error if parsing fails
    }
    
    res.status(400).json({
      success: false,
      message: errorMessage,
      error: process.env.NODE_ENV === 'development' 
        ? { original: error.message, data: errorData } 
        : undefined
    });
  }
};

/**
 * Cancel order
 * @route DELETE /api/delta/orders/:id
 * @access Private
 */
const cancelOrder = async (req, res) => {
  try {
    const { id } = req.params;
    const deltaApi = await getDeltaApiService(req.user);
    const data = await deltaApi.cancelOrder(id);
    
    res.status(200).json({
      success: true,
      data
    });
  } catch (error) {
    console.error('Cancel order error:', error);
    res.status(500).json({
      success: false,
      message: 'Error cancelling order',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Get market data
 * @route GET /api/delta/market-data
 * @access Private
 */
const getMarketData = async (req, res) => {
  try {
    // This is a custom endpoint that combines multiple Delta API calls
    // to provide comprehensive market data in a single request
    const { symbols } = req.query;
    
    if (!symbols) {
      return res.status(400).json({
        success: false,
        message: 'Symbols parameter is required'
      });
    }
    
    const symbolArray = symbols.split(',');
    const deltaApi = await getDeltaApiService(req.user);
    
    // Get products
    const products = await deltaApi.getProducts();
    
    // Filter products by symbols
    const filteredProducts = products.result.filter(
      product => symbolArray.includes(product.symbol)
    );
    
    // Get order books and recent trades for each product
    const marketData = await Promise.all(
      filteredProducts.map(async (product) => {
        try {
          const [orderBook, recentTrades] = await Promise.all([
            deltaApi.getOrderBook(product.id),
            deltaApi.getRecentTrades(product.id, 10)
          ]);
          
          return {
            product,
            orderBook: orderBook.result,
            recentTrades: recentTrades.result
          };
        } catch (error) {
          console.error(`Error fetching market data for ${product.symbol}:`, error);
          return {
            product,
            orderBook: { asks: [], bids: [] },
            recentTrades: []
          };
        }
      })
    );
    
    res.status(200).json({
      success: true,
      data: marketData
    });
  } catch (error) {
    console.error('Get market data error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching market data',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

module.exports = {
  getProducts,
  getOrderBook,
  getRecentTrades,
  getAccountBalance,
  getPositions,
  getOrders,
  createOrder,
  cancelOrder,
  getMarketData
}; 