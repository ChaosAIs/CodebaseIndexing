/**
 * A simple Node.js application demonstrating basic functionality
 */

const express = require('express');
const path = require('path');

class WebServer {
    constructor(port = 3000) {
        this.port = port;
        this.app = express();
        this.setupMiddleware();
        this.setupRoutes();
    }

    setupMiddleware() {
        this.app.use(express.json());
        this.app.use(express.static('public'));
    }

    setupRoutes() {
        this.app.get('/', this.handleHome.bind(this));
        this.app.get('/api/status', this.handleStatus.bind(this));
        this.app.post('/api/calculate', this.handleCalculate.bind(this));
    }

    handleHome(req, res) {
        res.send('<h1>Welcome to Demo App</h1>');
    }

    handleStatus(req, res) {
        res.json({
            status: 'running',
            timestamp: new Date().toISOString(),
            port: this.port
        });
    }

    handleCalculate(req, res) {
        const { operation, a, b } = req.body;
        
        let result;
        switch (operation) {
            case 'add':
                result = this.add(a, b);
                break;
            case 'multiply':
                result = this.multiply(a, b);
                break;
            case 'subtract':
                result = this.subtract(a, b);
                break;
            default:
                return res.status(400).json({ error: 'Invalid operation' });
        }
        
        res.json({ result, operation, a, b });
    }

    add(a, b) {
        return a + b;
    }

    multiply(a, b) {
        return a * b;
    }

    subtract(a, b) {
        return a - b;
    }

    start() {
        this.app.listen(this.port, () => {
            console.log(`Server running on port ${this.port}`);
        });
    }
}

function createServer(port) {
    return new WebServer(port);
}

function main() {
    const server = createServer(3001);
    server.start();
}

if (require.main === module) {
    main();
}

module.exports = { WebServer, createServer };
