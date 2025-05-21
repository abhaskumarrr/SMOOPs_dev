/**
 * Main Server Entry Point
 * Sets up and starts the Express server with API routes
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const path = require('path');
const dotenv = require('dotenv');
const http = require('http');
const { createWriteStream } = require('fs');
const prisma = require('./utils/prismaClient'); // Use the centralized prisma client
const { initializeWebsocketServer } = require('./sockets/websocketServer');

// Load environment variables
dotenv.config({
  path: path.resolve(__dirname, '../../.env')
});

// Create Express app
const app = express();
const PORT = process.env.PORT || 3001;
const NODE_ENV = process.env.NODE_ENV || 'development';

// Create HTTP server for Socket.IO
const server = http.createServer(app);

// Initialize WebSocket server
const io = initializeWebsocketServer(server);

// Logging middleware
const logStream = createWriteStream(path.join(__dirname, '../logs/server.log'), { flags: 'a' });
app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    const log = `${new Date().toISOString()} | ${req.method} ${req.url} ${res.statusCode} ${duration}ms\n`;
    
    logStream.write(log);
    
    if (NODE_ENV === 'development') {
      console.log(log);
    }
  });
  
  next();
});

// Security middleware
app.use(helmet());

// CORS middleware
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true
}));

// Body parser middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Set trust proxy if behind a proxy
if (process.env.TRUST_PROXY === 'true') {
  app.set('trust proxy', 1);
}

// Import routes
const healthRoutes = require('./routes/healthRoutes');
const authRoutes = require('./routes/authRoutes');
const userRoutes = require('./routes/userRoutes');
const apiKeyRoutes = require('./routes/apiKeyRoutes');
const deltaApiRoutes = require('./routes/deltaApiRoutes');
const botRoutes = require('./routes/botRoutes');

// Use routes
app.use('/api/health', healthRoutes);
app.use('/api/auth', authRoutes);
app.use('/api/users', userRoutes);
app.use('/api/keys', apiKeyRoutes);
app.use('/api/delta', deltaApiRoutes);
app.use('/api/bots', botRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  
  const statusCode = err.statusCode || 500;
  const message = err.message || 'Internal Server Error';
  
  res.status(statusCode).json({
    success: false,
    message,
    error: NODE_ENV === 'development' ? err.stack : undefined
  });
});

// Start server
server.listen(PORT, () => {
  console.log(`Server running in ${NODE_ENV} mode on port ${PORT}`);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
  console.error('Unhandled Promise Rejection:', err);
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
  process.exit(1);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down gracefully');
  
  // Close Prisma connection
  await prisma.$disconnect();
  
  // Close server
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
  
  // Force close after timeout
  setTimeout(() => {
    console.error('Could not close connections in time, forcefully shutting down');
    process.exit(1);
  }, 10000);
});

// Export for testing
module.exports = { app, server, io, prisma };

