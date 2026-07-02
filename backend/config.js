/**
 * Backend Configuration
 */

// Database configuration
export const DB_TYPE = 'sqlite'; // 'sqlite' or 'supabase'

// SQLite path (relative to backend directory)
export const SQLITE_PATH = './database.sqlite';

// Server configuration
export const config = {
  backendPort: parseInt(process.env.BACKEND_PORT) || 8787,
  corsOrigins: process.env.CORS_ORIGINS?.split(',') || [
    'http://localhost:8080',
    'http://127.0.0.1:8080',
  ],
};

// Supabase configuration (for future migration)
export const supabaseConfig = {
  url: process.env.SUPABASE_URL || 'https://your-project.supabase.co',
  key: process.env.SUPABASE_ANON_KEY || 'your-anon-key',
};

// Export adapter based on DB_TYPE
import { sqlite } from './db/sqlite.js';
// import { supabase } from './db/supabase.js'; // Uncomment when implemented

export const adapter = sqlite;
