/**
 * Utility functions for the demo application
 */

class Logger {
    constructor(name) {
        this.name = name;
        this.logs = [];
    }

    info(message) {
        const logEntry = `[INFO] ${this.name}: ${message}`;
        this.logs.push(logEntry);
        console.log(logEntry);
    }

    error(message) {
        const logEntry = `[ERROR] ${this.name}: ${message}`;
        this.logs.push(logEntry);
        console.error(logEntry);
    }

    getLogs() {
        return [...this.logs];
    }
}

function formatDate(date) {
    return date.toISOString().split('T')[0];
}

function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function generateId() {
    return Math.random().toString(36).substr(2, 9);
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

class DataStore {
    constructor() {
        this.data = new Map();
    }

    set(key, value) {
        this.data.set(key, value);
        return this;
    }

    get(key) {
        return this.data.get(key);
    }

    has(key) {
        return this.data.has(key);
    }

    delete(key) {
        return this.data.delete(key);
    }

    clear() {
        this.data.clear();
    }

    size() {
        return this.data.size;
    }

    keys() {
        return Array.from(this.data.keys());
    }

    values() {
        return Array.from(this.data.values());
    }
}

module.exports = {
    Logger,
    formatDate,
    validateEmail,
    generateId,
    delay,
    DataStore
};
