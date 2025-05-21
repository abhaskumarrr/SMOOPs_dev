/**
 * Centralized Prisma Client
 * Singleton instance with middleware and extensions applied
 */

const { PrismaClient } = require('../../generated/prisma');
const { withAccelerate } = require('@prisma/extension-accelerate');
const { applyMiddleware } = require('./prismaMiddleware');

// Check if we need to use Accelerate
const useAccelerate = process.env.USE_PRISMA_ACCELERATE === 'true';

// Create Prisma Client instance
const prismaBase = new PrismaClient({
  log: process.env.NODE_ENV === 'development' 
    ? ['query', 'info', 'warn', 'error'] 
    : ['warn', 'error'],
});

// Apply extensions conditionally
let prisma;
if (useAccelerate) {
  console.log('Using Prisma Accelerate for improved database performance');
  prisma = prismaBase.$extends(withAccelerate());
} else {
  prisma = prismaBase;
}

// Apply custom middleware
const prismaWithMiddleware = applyMiddleware(prisma);

// Add to global scope to prevent multiple instances in development
const globalForPrisma = global;
if (!globalForPrisma.prisma) {
  globalForPrisma.prisma = prismaWithMiddleware;
}

module.exports = globalForPrisma.prisma; 