/**
 * Delta Exchange API Service
 * Handles communication with Delta Exchange API (both testnet and mainnet)
 * 
 * References:
 * - Official Delta Exchange Documentation: https://docs.delta.exchange
 * - CCXT Delta Exchange Documentation: https://docs.ccxt.com/#/exchanges/delta
 */

const axios = require('axios');
const crypto = require('crypto');
const querystring = require('querystring');

// Get API key service for accessing stored keys
const apiKeyService = require('./apiKeyService');
const { createLogger } = require('../utils/logger');

// Create logger
const logger = createLogger('DeltaExchangeAPI');

// Environment configuration
const MAINNET_BASE_URL = 'https://api.delta.exchange';
const TESTNET_BASE_URL = 'https://testnet-api.delta.exchange';

// Default rate limit settings
const DEFAULT_RATE_LIMIT = {
  maxRetries: 3,
  initialDelay: 1000, // ms
  maxDelay: 10000, // ms
  factor: 2 // exponential backoff factor
};

/**
 * DeltaExchangeAPI Service
 * Provides methods to interact with Delta Exchange API
 */
class DeltaExchangeAPI {
  /**
   * Creates a new instance of the Delta Exchange API client
   * @param {Object} options - Configuration options
   * @param {boolean} options.testnet - Whether to use testnet (default: false)
   * @param {Object} options.rateLimit - Rate limit settings
   * @param {string} options.userId - User ID to retrieve API keys for
   */
  constructor(options = {}) {
    this.testnet = options.testnet || false;
    this.baseUrl = this.testnet ? TESTNET_BASE_URL : MAINNET_BASE_URL;
    this.rateLimit = { ...DEFAULT_RATE_LIMIT, ...(options.rateLimit || {}) };
    this.userId = options.userId;
    this.apiKeys = null;
    
    // Log initialization
    logger.info(`Initializing Delta Exchange API client with ${this.testnet ? 'testnet' : 'mainnet'} environment`);
  }

  /**
   * Initializes the API client with credentials
   * @param {Object} credentials - API credentials (optional, will use stored keys if not provided)
   * @param {string} credentials.apiKey - API key
   * @param {string} credentials.apiSecret - API secret
   */
  async initialize(credentials = null) {
    if (credentials) {
      this.apiKeys = credentials;
      logger.debug('Using provided API credentials');
    } else if (this.userId) {
      // Retrieve API keys from the secure storage
      logger.debug(`Retrieving API keys for user ${this.userId}`);
      this.apiKeys = await apiKeyService.getApiKey(this.userId);
      
      if (!this.apiKeys) {
        logger.error(`No API keys found for user ${this.userId}`);
        throw new Error('No API keys found for this user');
      }
    } else {
      logger.error('No credentials provided and no userId to retrieve keys');
      throw new Error('No credentials provided and no userId to retrieve keys');
    }

    // Set up axios instance with default configuration
    this.client = axios.create({
      baseURL: this.baseUrl,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Add response interceptor for logging
    this.client.interceptors.response.use(
      response => {
        this._logResponse(response);
        return response;
      },
      error => {
        this._logError(error);
        return Promise.reject(error);
      }
    );

    logger.info('Delta Exchange API client initialized successfully');
    return this;
  }

  /**
   * Gets server time from Delta Exchange
   * @returns {Promise<Object>} Server time information
   */
  async getServerTime() {
    return this._makeRequest({
      method: 'GET',
      endpoint: '/v2/time'
    });
  }

  /**
   * Gets all available markets from Delta Exchange
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Available markets
   */
  async getMarkets(params = {}) {
    return this._makeRequest({
      method: 'GET',
      endpoint: '/v2/products',
      params
    });
  }

  /**
   * Gets market data for a specific symbol
   * @param {string} symbol - Market symbol (e.g., 'BTCUSD')
   * @returns {Promise<Object>} Market data
   */
  async getMarketData(symbol) {
    return this._makeRequest({
      method: 'GET',
      endpoint: `/v2/products/${symbol}`
    });
  }

  /**
   * Gets ticker information for a specific symbol
   * @param {string} symbol - Market symbol (e.g., 'BTCUSD')
   * @returns {Promise<Object>} Ticker data
   */
  async getTicker(symbol) {
    return this._makeRequest({
      method: 'GET',
      endpoint: `/v2/tickers`,
      params: { symbol }
    });
  }

  /**
   * Gets orderbook for a specific symbol
   * @param {string} symbol - Market symbol (e.g., 'BTCUSD')
   * @param {number} depth - Orderbook depth (default: 10)
   * @returns {Promise<Object>} Orderbook data
   */
  async getOrderbook(symbol, depth = 10) {
    return this._makeRequest({
      method: 'GET',
      endpoint: `/v2/l2orderbook/${symbol}`,
      params: { depth }
    });
  }

  /**
   * Gets the user's account information
   * @returns {Promise<Object>} Account information
   */
  async getAccountInfo() {
    return this._makeRequest({
      method: 'GET',
      endpoint: '/v2/user',
      authenticated: true
    });
  }

  /**
   * Gets the user's wallet balances
   * @returns {Promise<Object>} Wallet balances
   */
  async getWalletBalances() {
    return this._makeRequest({
      method: 'GET',
      endpoint: '/v2/wallet/balances',
      authenticated: true
    });
  }

  /**
   * Gets the user's active positions
   * @returns {Promise<Object>} Active positions
   */
  async getPositions() {
    return this._makeRequest({
      method: 'GET',
      endpoint: '/v2/positions',
      authenticated: true
    });
  }

  /**
   * Gets the user's active orders
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Active orders
   */
  async getActiveOrders(params = {}) {
    return this._makeRequest({
      method: 'GET',
      endpoint: '/v2/orders',
      params,
      authenticated: true
    });
  }

  /**
   * Places a new order
   * @param {Object} order - Order details
   * @param {string} order.symbol - Market symbol (e.g., 'BTCUSD')
   * @param {string} order.side - Order side ('buy' or 'sell')
   * @param {string} order.type - Order type ('limit', 'market', etc.)
   * @param {number} order.size - Order size
   * @param {number} order.price - Order price (required for limit orders)
   * @param {string} order.timeInForce - Time in force ('gtc', 'ioc', 'fok')
   * @returns {Promise<Object>} Order information
   */
  async placeOrder(order) {
    // Basic validation
    if (!order.symbol) throw new Error('Symbol is required');
    if (!order.side) throw new Error('Side is required');
    if (!order.size) throw new Error('Size is required');
    if (order.type === 'limit' && !order.price) throw new Error('Price is required for limit orders');

    const payload = {
      symbol: order.symbol,
      side: order.side.toUpperCase(),
      size: order.size,
      order_type: order.type || 'limit',
      time_in_force: order.timeInForce || 'gtc'
    };

    if (order.price) payload.price = order.price;
    if (order.reduceOnly) payload.reduce_only = order.reduceOnly;
    if (order.postOnly) payload.post_only = order.postOnly;
    if (order.clientOrderId) payload.client_order_id = order.clientOrderId;

    logger.info(`Placing ${order.side} order for ${order.size} ${order.symbol}`);
    
    return this._makeRequest({
      method: 'POST',
      endpoint: '/v2/orders',
      data: payload,
      authenticated: true
    });
  }

  /**
   * Cancels an order
   * @param {string} orderId - Order ID to cancel
   * @returns {Promise<Object>} Cancellation response
   */
  async cancelOrder(orderId) {
    logger.info(`Cancelling order ${orderId}`);
    
    return this._makeRequest({
      method: 'DELETE',
      endpoint: `/v2/orders/${orderId}`,
      authenticated: true
    });
  }

  /**
   * Cancels all active orders
   * @param {Object} params - Filter parameters
   * @returns {Promise<Object>} Cancellation response
   */
  async cancelAllOrders(params = {}) {
    logger.info('Cancelling all active orders', params);
    
    return this._makeRequest({
      method: 'DELETE',
      endpoint: '/v2/orders',
      params,
      authenticated: true
    });
  }

  /**
   * Gets order history for the user
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Order history
   */
  async getOrderHistory(params = {}) {
    return this._makeRequest({
      method: 'GET',
      endpoint: '/v2/orders/history',
      params,
      authenticated: true
    });
  }

  /**
   * Gets trade history for the user
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Trade history
   */
  async getTradeHistory(params = {}) {
    return this._makeRequest({
      method: 'GET',
      endpoint: '/v2/fills',
      params,
      authenticated: true
    });
  }

  /**
   * Makes a request to the Delta Exchange API with retries and rate limit handling
   * @private
   * @param {Object} options - Request options
   * @param {string} options.method - HTTP method
   * @param {string} options.endpoint - API endpoint
   * @param {Object} options.params - Query parameters
   * @param {Object} options.data - Request data
   * @param {boolean} options.authenticated - Whether request requires authentication
   * @returns {Promise<Object>} API response
   */
  async _makeRequest(options, retryCount = 0) {
    const { method, endpoint, params, data, authenticated } = options;
    
    try {
      // Prepare request config
      const requestConfig = {
        method,
        url: endpoint
      };

      // Add query parameters if provided
      if (params) {
        requestConfig.params = params;
      }

      // Add data if provided
      if (data) {
        requestConfig.data = data;
      }

      // Add authentication if required
      if (authenticated) {
        if (!this.apiKeys) {
          logger.error('API keys not initialized');
          throw new Error('API keys not initialized');
        }
        this._addAuthHeaders(requestConfig);
      }

      // Log the request
      this._logRequest(requestConfig);

      // Make the request
      const response = await this.client(requestConfig);
      return response.data;
    } catch (error) {
      // Handle rate limiting errors
      if (error.response && error.response.status === 429) {
        // Rate limit exceeded
        if (retryCount < this.rateLimit.maxRetries) {
          // Calculate delay with exponential backoff
          const delay = Math.min(
            this.rateLimit.initialDelay * Math.pow(this.rateLimit.factor, retryCount),
            this.rateLimit.maxDelay
          );
          
          // Log the retry
          logger.warn(`Rate limit exceeded. Retrying in ${delay}ms (attempt ${retryCount + 1}/${this.rateLimit.maxRetries})`);
          
          // Wait and retry
          await new Promise(resolve => setTimeout(resolve, delay));
          return this._makeRequest(options, retryCount + 1);
        }
      }
      
      // Handle other errors
      if (error.response) {
        logger.error(`API Error: ${error.response.status}`, error.response.data);
        throw new Error(`Delta Exchange API Error: ${error.response.status} - ${JSON.stringify(error.response.data)}`);
      } else if (error.request) {
        logger.error('Request Error', error.message);
        throw new Error(`Delta Exchange API Request Error: ${error.message}`);
      } else {
        logger.error('Error', error);
        throw error;
      }
    }
  }

  /**
   * Adds authentication headers to a request
   * @private
   * @param {Object} requestConfig - Axios request configuration
   */
  _addAuthHeaders(requestConfig) {
    const timestamp = Math.floor(Date.now());
    const method = requestConfig.method.toUpperCase();
    const path = requestConfig.url;
    
    // Prepare the message to sign
    let message = timestamp + method + path;
    
    // Add params to the message if they exist
    if (requestConfig.params) {
      const queryString = querystring.stringify(requestConfig.params);
      if (queryString) {
        message += '?' + queryString;
      }
    }
    
    // Add body to the message if it exists
    if (requestConfig.data) {
      message += JSON.stringify(requestConfig.data);
    }
    
    // Create the signature
    const signature = crypto
      .createHmac('sha256', this.apiKeys.secret)
      .update(message)
      .digest('hex');
    
    // Add authentication headers
    requestConfig.headers = {
      ...requestConfig.headers,
      'api-key': this.apiKeys.key,
      'timestamp': timestamp.toString(),
      'signature': signature
    };
    
    logger.debug('Added authentication headers');
  }

  /**
   * Logs a request
   * @private
   * @param {Object} request - Request configuration
   */
  _logRequest(request) {
    // Create a safe copy for logging (remove sensitive data)
    const safeRequest = JSON.parse(JSON.stringify(request));
    
    if (safeRequest.headers && safeRequest.headers['api-key']) {
      safeRequest.headers['api-key'] = '***';
      safeRequest.headers['signature'] = '***';
    }
    
    logger.info(`API Request: ${request.method} ${request.url}`);
    logger.debug('Request details', safeRequest);
  }

  /**
   * Logs a response
   * @private
   * @param {Object} response - Axios response
   */
  _logResponse(response) {
    logger.info(`API Response (${response.status}): ${response.config.method} ${response.config.url}`);
    
    // Log response data in debug mode
    if (response.data) {
      logger.debug('Response data', {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
        // Only log a sample of the data for large responses
        dataSample: typeof response.data === 'object' ? 
          JSON.stringify(response.data).substring(0, 200) + '...' : 
          response.data
      });
    }
  }

  /**
   * Logs an error
   * @private
   * @param {Object} error - Axios error
   */
  _logError(error) {
    if (error.response) {
      // The request was made and the server responded with a status code outside of 2xx
      logger.error(
        `API Error (${error.response.status}): ${error.config.method} ${error.config.url}`,
        error.response.data
      );
    } else if (error.request) {
      // The request was made but no response was received
      logger.error('Request Error', { message: error.message, request: error.request });
    } else {
      // Something happened in setting up the request
      logger.error('Config Error', error.message);
    }
  }
}

module.exports = DeltaExchangeAPI; 