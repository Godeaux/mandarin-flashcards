/**
 * SQLite Database Implementation
 *
 * Uses a file-based SQLite database. Perfect for self-hosting on Mac Studio.
 * To switch to Supabase, replace this file with supabase.js
 */

import sqlite3 from 'sqlite3';
import { DatabaseAdapter } from './adapter.js';

export class SQLiteAdapter extends DatabaseAdapter {
  constructor(dbPath = 'database.sqlite') {
    super();
    this.dbPath = dbPath;
    this.db = null;
  }

  async init() {
    return new Promise((resolve, reject) => {
      this.db = new sqlite3.Database(this.dbPath, (err) => {
        if (err) {
          reject(err);
          return;
        }
        this.db.configure('busyTimeout', 5000);
        this.initSchema();
        resolve();
      });
    });
  }

  initSchema() {
    // Users table
    this.db.run(`
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        license_key TEXT UNIQUE
      )
    `);

    // Progress table
    this.db.run(`
      CREATE TABLE IF NOT EXISTS progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        card_id TEXT NOT NULL,
        srs_data JSON NOT NULL,
        updated_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(id),
        UNIQUE(user_id, card_id)
      )
    `);

    // Sessions table
    this.db.run(`
      CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_data JSON NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(id)
      )
    `);

    // Streaks table
    this.db.run(`
      CREATE TABLE IF NOT EXISTS streaks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        streak_data JSON NOT NULL,
        updated_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(id)
      )
    `);

    // License keys table
    this.db.run(`
      CREATE TABLE IF NOT EXISTS license_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE NOT NULL,
        user_id INTEGER,
        created_at TEXT DEFAULT (datetime('now')),
        is_used INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users(id)
      )
    `);
  }

  async close() {
    return new Promise((resolve, reject) => {
      if (this.db) {
        this.db.close((err) => {
          if (err) reject(err);
          else resolve();
        });
      } else {
        resolve();
      }
    });
  }

  // --- User Operations ---

  async createUser(email, passwordHash) {
    return new Promise((resolve, reject) => {
      this.db.run(
        'INSERT INTO users (email, password_hash) VALUES (?, ?)',
        [email, passwordHash],
        function (err) {
          if (err) reject(err);
          else resolve({ id: this.lastID, email });
        }
      );
    });
  }

  async findUserByEmail(email) {
    return new Promise((resolve, reject) => {
      this.db.get('SELECT * FROM users WHERE email = ?', [email], (err, row) => {
        if (err) reject(err);
        else resolve(row || null);
      });
    });
  }

  async findUserByLicenseKey(licenseKey) {
    return new Promise((resolve, reject) => {
      this.db.get('SELECT * FROM users WHERE license_key = ?', [licenseKey], (err, row) => {
        if (err) reject(err);
        else resolve(row || null);
      });
    });
  }

  async setUserLicenseKey(userId, licenseKey) {
    return new Promise((resolve, reject) => {
      this.db.run('UPDATE users SET license_key = ? WHERE id = ?', [licenseKey, userId], (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }

  // --- Progress Operations ---

  async getProgress(userId, cardId) {
    return new Promise((resolve, reject) => {
      this.db.get(
        'SELECT * FROM progress WHERE user_id = ? AND card_id = ?',
        [userId, cardId],
        (err, row) => {
          if (err) reject(err);
          else resolve(row || null);
        }
      );
    });
  }

  async saveProgress(userId, cardId, srsData) {
    const progress = await this.getProgress(userId, cardId);

    return new Promise((resolve, reject) => {
      if (progress) {
        this.db.run(
          'UPDATE progress SET srs_data = ?, updated_at = datetime("now") WHERE id = ?',
          [JSON.stringify(srsData), progress.id],
          (err) => {
            if (err) reject(err);
            else resolve();
          }
        );
      } else {
        this.db.run(
          'INSERT INTO progress (user_id, card_id, srs_data) VALUES (?, ?, ?)',
          [userId, cardId, JSON.stringify(srsData)],
          (err) => {
            if (err) reject(err);
            else resolve();
          }
        );
      }
    });
  }

  async getAllProgress(userId) {
    return new Promise((resolve, reject) => {
      this.db.all('SELECT * FROM progress WHERE user_id = ?', [userId], (err, rows) => {
        if (err) reject(err);
        else resolve(rows || []);
      });
    });
  }

  // --- Session/Stats Operations ---

  async saveSession(userId, sessionData) {
    return new Promise((resolve, reject) => {
      this.db.run(
        'INSERT INTO sessions (user_id, session_data) VALUES (?, ?)',
        [userId, JSON.stringify(sessionData)],
        (err) => {
          if (err) reject(err);
          else resolve();
        }
      );
    });
  }

  async getSession(userId) {
    return new Promise((resolve, reject) => {
      this.db.get(
        'SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1',
        [userId],
        (err, row) => {
          if (err) reject(err);
          else resolve(row ? JSON.parse(row.session_data) : null);
        }
      );
    });
  }

  async saveStreak(userId, streakData) {
    const existing = await this.getStreak(userId);

    return new Promise((resolve, reject) => {
      if (existing) {
        this.db.run(
          'UPDATE streaks SET streak_data = ?, updated_at = datetime("now") WHERE id = ?',
          [JSON.stringify(streakData), existing.id],
          (err) => {
            if (err) reject(err);
            else resolve();
          }
        );
      } else {
        this.db.run(
          'INSERT INTO streaks (user_id, streak_data) VALUES (?, ?)',
          [userId, JSON.stringify(streakData)],
          (err) => {
            if (err) reject(err);
            else resolve();
          }
        );
      }
    });
  }

  async getStreak(userId) {
    return new Promise((resolve, reject) => {
      this.db.get(
        'SELECT * FROM streaks WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1',
        [userId],
        (err, row) => {
          if (err) reject(err);
          else resolve(row ? JSON.parse(row.streak_data) : null);
        }
      );
    });
  }

  // --- License Keys ---

  async generateLicenseKey() {
    const crypto = await import('crypto');
    return new Promise((resolve, reject) => {
      // Generate a license key: XXXX-XXXX-XXXX-XXXX
      const generateKey = () => {
        const parts = [];
        for (let i = 0; i < 4; i++) {
          parts.push(crypto.randomBytes(2).toString('hex').toUpperCase());
        }
        return parts.join('-');
      };

      const key = generateKey();

      this.db.get('SELECT 1 FROM license_keys WHERE key = ?', [key], (err, row) => {
        if (err) reject(err);
        else if (row)
          resolve(generateKey()); // Key exists, try again
        else resolve(key);
      });
    });
  }

  async useLicenseKey(licenseKey, userId) {
    return new Promise((resolve, reject) => {
      this.db.run(
        'UPDATE license_keys SET is_used = 1, user_id = ? WHERE key = ?',
        [userId, licenseKey],
        (err) => {
          if (err) reject(err);
          else resolve();
        }
      );
    });
  }
}

export const sqlite = new SQLiteAdapter();
