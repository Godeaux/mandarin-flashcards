/**
 * Supabase Database Implementation
 *
 * This file can be used to switch from SQLite to Supabase when ready.
 * To use:
 *   1. Uncomment the import in config.js
 *   2. Set DB_TYPE = 'supabase'
 *   3. Update supabaseConfig with your Supabase project credentials
 */

import { createClient } from '@supabase/supabase-js';
import { supabaseConfig } from '../config.js';
import { DatabaseAdapter } from './adapter.js';

export class SupabaseAdapter extends DatabaseAdapter {
  constructor() {
    super();
    this.supabase = null;
  }

  async init() {
    if (!this.supabase) {
      this.supabase = createClient(supabaseConfig.url, supabaseConfig.key);
    }
    return Promise.resolve();
  }

  async close() {
    // Supabase client doesn't need explicit close
    return Promise.resolve();
  }

  // --- User Operations ---

  async createUser(email, passwordHash) {
    const { data, error } = await this.supabase
      .from('users')
      .insert([{ email, password_hash: passwordHash }])
      .select()
      .single();

    if (error) throw error;
    return { id: data.id, email: data.email };
  }

  async findUserByEmail(email) {
    const { data, error } = await this.supabase
      .from('users')
      .select('*')
      .eq('email', email)
      .single();

    if (error && error.code !== 'PGRST116') throw error;
    return data || null;
  }

  async findUserByLicenseKey(licenseKey) {
    const { data, error } = await this.supabase
      .from('users')
      .select('*')
      .eq('license_key', licenseKey)
      .single();

    if (error && error.code !== 'PGRST116') throw error;
    return data || null;
  }

  async setUserLicenseKey(userId, licenseKey) {
    const { error } = await this.supabase
      .from('users')
      .update({ license_key: licenseKey })
      .eq('id', userId);

    if (error) throw error;
  }

  // --- Progress Operations ---

  async getProgress(userId, cardId) {
    const { data, error } = await this.supabase
      .from('progress')
      .select('*')
      .eq('user_id', userId)
      .eq('card_id', cardId)
      .single();

    if (error && error.code !== 'PGRST116') throw error;
    return data || null;
  }

  async saveProgress(userId, cardId, srsData) {
    const progress = await this.getProgress(userId, cardId);

    if (progress) {
      const { error } = await this.supabase
        .from('progress')
        .update({ srs_data: srsData })
        .eq('id', progress.id);
      if (error) throw error;
    } else {
      const { error } = await this.supabase
        .from('progress')
        .insert([{ user_id: userId, card_id: cardId, srs_data: srsData }]);
      if (error) throw error;
    }
  }

  async getAllProgress(userId) {
    const { data, error } = await this.supabase.from('progress').select('*').eq('user_id', userId);

    if (error) throw error;
    return data || [];
  }

  // --- Session/Stats Operations ---

  async saveSession(userId, sessionData) {
    const { error } = await this.supabase
      .from('sessions')
      .insert([{ user_id: userId, session_data: sessionData }]);
    if (error) throw error;
  }

  async getSession(userId) {
    const { data, error } = await this.supabase
      .from('sessions')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    if (error && error.code !== 'PGRST116') throw error;
    return data ? JSON.parse(data.session_data) : null;
  }

  async saveStreak(userId, streakData) {
    const existing = await this.getStreak(userId);

    if (existing) {
      const { error } = await this.supabase
        .from('streaks')
        .update({ streak_data: streakData })
        .eq('id', existing.id);
      if (error) throw error;
    } else {
      const { error } = await this.supabase
        .from('streaks')
        .insert([{ user_id: userId, streak_data: streakData }]);
      if (error) throw error;
    }
  }

  async getStreak(userId) {
    const { data, error } = await this.supabase
      .from('streaks')
      .select('*')
      .eq('user_id', userId)
      .order('updated_at', { ascending: false })
      .limit(1)
      .single();

    if (error && error.code !== 'PGRST116') throw error;
    return data ? JSON.parse(data.streak_data) : null;
  }

  // --- License Keys ---

  async generateLicenseKey() {
    const crypto = await import('crypto');

    const generateKey = () => {
      const parts = [];
      for (let i = 0; i < 4; i++) {
        parts.push(crypto.randomBytes(2).toString('hex').toUpperCase());
      }
      return parts.join('-');
    };

    const key = generateKey();

    // Check if key exists
    const { data } = await this.supabase.from('license_keys').select('id').eq('key', key).single();

    if (data) return generateKey(); // Key exists, try again
    return key;
  }

  async useLicenseKey(licenseKey, userId) {
    const { error } = await this.supabase
      .from('license_keys')
      .update({ is_used: 1, user_id: userId })
      .eq('key', licenseKey);

    if (error) throw error;
  }
}

export const supabase = new SupabaseAdapter();
