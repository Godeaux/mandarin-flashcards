/**
 * Database Adapter Interface
 *
 * This module defines the interface for database operations.
 * Two implementations:
 *   - sqlite.js: SQLite (file-based, self-hosted)
 *   - supabase.js: Supabase (cloud database)
 */

export class DatabaseAdapter {
  /**
   * Initialize the database connection
   */
  async init() {
    throw new Error('init() must be implemented');
  }

  /**
   * Close the database connection
   */
  async close() {
    throw new Error('close() must be implemented');
  }

  // --- User Operations ---

  /**
   * Create a new user
   * @param {string} email - User's email
   * @param {string} passwordHash - Hashed password
   * @returns {Promise<object>} Created user with id
   */
  async createUser(email, passwordHash) {
    throw new Error('createUser() must be implemented');
  }

  /**
   * Find user by email
   * @param {string} email - User's email
   * @returns {Promise<object|null>} User object or null
   */
  async findUserByEmail(email) {
    throw new Error('findUserByEmail() must be implemented');
  }

  /**
   * Find user by license key
   * @param {string} licenseKey - License key
   * @returns {Promise<object|null>} User object or null
   */
  async findUserByLicenseKey(licenseKey) {
    throw new Error('findUserByLicenseKey() must be implemented');
  }

  /**
   * Update user's license key
   * @param {number} userId - User ID
   * @param {string} licenseKey - License key to set
   */
  async setUserLicenseKey(userId, licenseKey) {
    throw new Error('setUserLicenseKey() must be implemented');
  }

  // --- Progress Operations ---

  /**
   * Get user's SRS progress for a card
   * @param {number} userId - User ID
   * @param {string} cardId - Card ID
   * @returns {Promise<object|null>} Progress object or null
   */
  async getProgress(userId, cardId) {
    throw new Error('getProgress() must be implemented');
  }

  /**
   * Save or update user's SRS progress for a card
   * @param {number} userId - User ID
   * @param {string} cardId - Card ID
   * @param {object} srsData - SRS state data
   */
  async saveProgress(userId, cardId, srsData) {
    throw new Error('saveProgress() must be implemented');
  }

  /**
   * Get all progress for a user
   * @param {number} userId - User ID
   * @returns {Promise<object[]>} Array of progress objects
   */
  async getAllProgress(userId) {
    throw new Error('getAllProgress() must be implemented');
  }

  // --- Session/Stats Operations ---

  /**
   * Save user's session data
   * @param {number} userId - User ID
   * @param {object} sessionData - Session data
   */
  async saveSession(userId, sessionData) {
    throw new Error('saveSession() must be implemented');
  }

  /**
   * Get user's session data
   * @param {number} userId - User ID
   * @returns {Promise<object|null>} Session data or null
   */
  async getSession(userId) {
    throw new Error('getSession() must be implemented');
  }

  /**
   * Save user's streak data
   * @param {number} userId - User ID
   * @param {object} streakData - Streak data
   */
  async saveStreak(userId, streakData) {
    throw new Error('saveStreak() must be implemented');
  }

  /**
   * Get user's streak data
   * @param {number} userId - User ID
   * @returns {Promise<object|null>} Streak data or null
   */
  async getStreak(userId) {
    throw new Error('getStreak() must be implemented');
  }

  // --- License Keys ---

  /**
   * Generate a new license key
   * @returns {Promise<string>} Generated license key
   */
  async generateLicenseKey() {
    throw new Error('generateLicenseKey() must be implemented');
  }

  /**
   * Mark a license key as used for a user
   * @param {string} licenseKey - License key
   * @param {number} userId - User ID
   */
  async useLicenseKey(licenseKey, userId) {
    throw new Error('useLicenseKey() must be implemented');
  }
}
