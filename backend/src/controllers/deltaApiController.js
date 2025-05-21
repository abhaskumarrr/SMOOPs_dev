/**
 * Delta Exchange API Controller
 * Handles HTTP requests related to Delta Exchange API operations
 */

const DeltaExchangeAPI = require('../services/deltaApiService');

/**
 * Gets market data
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
async function getMarketData(req, res) {
  try {
    const { symbol } = req.params;
    const userId = req.user.id; // From auth middleware
    
    if (!symbol) {
      return res.status(400).json({
        success: false,
        message: 'Symbol is required'
      });
    }
    
    // Create and initialize API client
    const deltaApi = new DeltaExchangeAPI({ 
      userId,
      testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true'
    });
    
    await deltaApi.initialize();
    
    // Get market data
    const marketData = await deltaApi.getMarketData(symbol);
    
    return res.status(200).json({
      success: true,
      data: marketData
    });
  } catch (error) {
    console.error('Error getting market data:', error.message);
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch market data',
      error: error.message
    });
  }
}

/**
 * Gets all available markets
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
async function getMarkets(req, res) {
  try {
    // Create API client (doesn't need authentication for public endpoints)
    const deltaApi = new DeltaExchangeAPI({ 
      testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true'
    });
    
    await deltaApi.initialize();
    
    // Get markets with any query parameters
    const markets = await deltaApi.getMarkets(req.query);
    
    return res.status(200).json({
      success: true,
      data: markets
    });
  } catch (error) {
    console.error('Error getting markets:', error.message);
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch markets',
      error: error.message
    });
  }
}

/**
 * Places an order
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
async function placeOrder(req, res) {
  try {
    const { symbol, side, type, size, price, timeInForce, reduceOnly, postOnly, clientOrderId } = req.body;
    const userId = req.user.id; // From auth middleware
    
    // Validate required fields
    if (!symbol || !side || !size) {
      return res.status(400).json({
        success: false,
        message: 'Symbol, side, and size are required'
      });
    }
    
    // Validate limit orders must have price
    if ((type === 'limit' || !type) && !price) {
      return res.status(400).json({
        success: false,
        message: 'Price is required for limit orders'
      });
    }
    
    // Create and initialize API client
    const deltaApi = new DeltaExchangeAPI({ 
      userId,
      testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true'
    });
    
    await deltaApi.initialize();
    
    // Prepare order object
    const order = {
      symbol,
      side,
      type,
      size,
      price,
      timeInForce,
      reduceOnly,
      postOnly,
      clientOrderId
    };
    
    // Place order
    const result = await deltaApi.placeOrder(order);
    
    return res.status(201).json({
      success: true,
      data: result
    });
  } catch (error) {
    console.error('Error placing order:', error.message);
    return res.status(500).json({
      success: false,
      message: 'Failed to place order',
      error: error.message
    });
  }
}

/**
 * Cancels an order
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
async function cancelOrder(req, res) {
  try {
    const { orderId } = req.params;
    const userId = req.user.id; // From auth middleware
    
    if (!orderId) {
      return res.status(400).json({
        success: false,
        message: 'Order ID is required'
      });
    }
    
    // Create and initialize API client
    const deltaApi = new DeltaExchangeAPI({ 
      userId,
      testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true'
    });
    
    await deltaApi.initialize();
    
    // Cancel order
    const result = await deltaApi.cancelOrder(orderId);
    
    return res.status(200).json({
      success: true,
      data: result
    });
  } catch (error) {
    console.error('Error canceling order:', error.message);
    return res.status(500).json({
      success: false,
      message: 'Failed to cancel order',
      error: error.message
    });
  }
}

/**
 * Gets active orders
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
async function getActiveOrders(req, res) {
  try {
    const userId = req.user.id; // From auth middleware
    
    // Create and initialize API client
    const deltaApi = new DeltaExchangeAPI({ 
      userId,
      testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true'
    });
    
    await deltaApi.initialize();
    
    // Get active orders with any query parameters
    const orders = await deltaApi.getActiveOrders(req.query);
    
    return res.status(200).json({
      success: true,
      data: orders
    });
  } catch (error) {
    console.error('Error getting active orders:', error.message);
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch active orders',
      error: error.message
    });
  }
}

/**
 * Gets user wallet balances
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
async function getWalletBalances(req, res) {
  try {
    const userId = req.user.id; // From auth middleware
    
    // Create and initialize API client
    const deltaApi = new DeltaExchangeAPI({ 
      userId,
      testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true'
    });
    
    await deltaApi.initialize();
    
    // Get wallet balances
    const balances = await deltaApi.getWalletBalances();
    
    return res.status(200).json({
      success: true,
      data: balances
    });
  } catch (error) {
    console.error('Error getting wallet balances:', error.message);
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch wallet balances',
      error: error.message
    });
  }
}

/**
 * Gets user positions
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
async function getPositions(req, res) {
  try {
    const userId = req.user.id; // From auth middleware
    
    // Create and initialize API client
    const deltaApi = new DeltaExchangeAPI({ 
      userId,
      testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true'
    });
    
    await deltaApi.initialize();
    
    // Get positions
    const positions = await deltaApi.getPositions();
    
    return res.status(200).json({
      success: true,
      data: positions
    });
  } catch (error) {
    console.error('Error getting positions:', error.message);
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch positions',
      error: error.message
    });
  }
}

module.exports = {
  getMarketData,
  getMarkets,
  placeOrder,
  cancelOrder,
  getActiveOrders,
  getWalletBalances,
  getPositions
}; 