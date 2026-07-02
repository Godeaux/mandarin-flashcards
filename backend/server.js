/**
 * 漢字 Study Backend
 *
 * A Node.js/Express backend for user authentication, progress sync, and license validation.
 * Uses SQLite by default (self-hosted), but can be swapped for Supabase.
 */

import express from 'express';
import cors from 'cors';
import { config, adapter } from './config.js';

const app = express();
app.use(cors());
app.use(express.json());

// Middleware to extract user from JWT
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  // Token validation would go here
  // For now, we'll get user_id from token payload
  const userId = parseInt(token.split('|')[0]) || 1; // Default to user 1 for development
  req.user = { id: userId };
  next();
};

// ==========================================
// Auth Routes
// ==========================================

app.post('/api/auth/register', async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }

    // Check if user exists
    const existingUser = await adapter.findUserByEmail(email);
    if (existingUser) {
      return res.status(409).json({ error: 'Email already registered' });
    }

    // Hash password (in production, use proper hashing)
    const passwordHash = password; // Replace with bcrypt.hash(password) in production

    // Create user
    const user = await adapter.createUser(email, passwordHash);

    res.status(201).json({
      message: 'User created successfully',
      user: { id: user.id, email: user.email },
    });
  } catch (err) {
    console.error('Registration error:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }

    const user = await adapter.findUserByEmail(email);
    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Verify password (replace with bcrypt.compare in production)
    if (password !== user.password_hash) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Generate token (in production, use JWT properly)
    const token = `${user.id}|${Date.now()}`;

    res.json({
      message: 'Login successful',
      token,
      user: { id: user.id, email: user.email, licenseKey: user.license_key },
    });
  } catch (err) {
    console.error('Login error:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.post('/api/auth/verify-license', async (req, res) => {
  try {
    const { licenseKey } = req.body;

    if (!licenseKey) {
      return res.status(400).json({ error: 'License key is required' });
    }

    const user = await adapter.findUserByLicenseKey(licenseKey);

    if (user) {
      res.json({
        valid: true,
        userId: user.id,
        email: user.email,
        licenseKey: user.license_key,
      });
    } else {
      res.json({ valid: false });
    }
  } catch (err) {
    console.error('License verification error:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

// ==========================================
// Progress Sync Routes
// ==========================================

app.get('/api/progress', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;

    // Get all progress for this user
    const progressList = await adapter.getAllProgress(userId);

    // Convert to object keyed by card_id
    const progress = {};
    for (const item of progressList) {
      try {
        progress[item.card_id] = JSON.parse(item.srs_data);
      } catch (e) {
        progress[item.card_id] = item.srs_data;
      }
    }

    res.json({ progress });
  } catch (err) {
    console.error('Get progress error:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.post('/api/progress', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const { srsData, session, streak } = req.body;

    // Save SRS progress for each card
    if (srsData) {
      for (const [cardId, srs] of Object.entries(srsData)) {
        await adapter.saveProgress(userId, cardId, srs);
      }
    }

    // Save session and streak
    if (session) {
      await adapter.saveSession(userId, session);
    }
    if (streak) {
      await adapter.saveStreak(userId, streak);
    }

    res.json({ message: 'Progress saved successfully' });
  } catch (err) {
    console.error('Save progress error:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

// ==========================================
// License Key Routes
// ==========================================

app.get('/api/license/key', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;

    // Generate a new license key
    const key = await adapter.generateLicenseKey();

    res.json({ licenseKey: key });
  } catch (err) {
    console.error('Generate license key error:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.post('/api/license/use', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const { licenseKey } = req.body;

    if (!licenseKey) {
      return res.status(400).json({ error: 'License key is required' });
    }

    // Mark the key as used
    await adapter.useLicenseKey(licenseKey, userId);

    res.json({ message: 'License key activated successfully' });
  } catch (err) {
    console.error('Use license key error:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

// ==========================================
// Status Check
// ==========================================

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// ==========================================
// Start Server
// ==========================================

const PORT = config.backendPort || 8787;

adapter
  .init()
  .then(() => {
    app.listen(PORT, '0.0.0.0', () => {
      console.log(`Backend server running on port ${PORT}`);
      console.log('Database initialized successfully');
    });
  })
  .catch((err) => {
    console.error('Failed to initialize database:', err);
    process.exit(1);
  });

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('Shutting down gracefully...');
  await adapter.close();
  process.exit(0);
});

export { app };
