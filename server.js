const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// In-memory storage for demo purposes
let chatHistory = [];
let appointments = [];
let contacts = [];

// API Routes

// Chat endpoint
app.post('/api/chat', (req, res) => {
  const { message } = req.body;
  
  if (!message) {
    return res.status(400).json({ error: 'Message is required' });
  }

  console.log('User message:', message);
  
  // Store user message
  chatHistory.push({ type: 'user', message, timestamp: new Date() });
  
  // Generate AI response (hardcoded for demo)
  const aiResponses = [
    "Thank you for sharing that with me. I'm here to listen and support you.",
    "I understand how you're feeling. Remember, it's okay to have these emotions.",
    "That sounds challenging. Would you like to talk more about what's on your mind?",
    "I appreciate you opening up to me. You're not alone in this.",
    "Your feelings are valid. Let's work through this together.",
    "I'm here to help you process these thoughts. What would be most helpful right now?"
  ];
  
  const randomResponse = aiResponses[Math.floor(Math.random() * aiResponses.length)];
  
  // Store AI response
  chatHistory.push({ type: 'ai', message: randomResponse, timestamp: new Date() });
  
  res.json({ 
    success: true, 
    response: randomResponse,
    timestamp: new Date()
  });
});

// Appointments endpoint
app.post('/api/appointments', (req, res) => {
  const { date, time, name, email } = req.body;
  
  if (!date) {
    return res.status(400).json({ error: 'Date is required' });
  }
  
  const appointment = {
    id: appointments.length + 1,
    date,
    time: time || 'TBD',
    name: name || 'Anonymous',
    email: email || 'Not provided',
    createdAt: new Date()
  };
  
  appointments.push(appointment);
  
  console.log(`Appointment booked for ${date} at ${time || 'TBD'} by ${name || 'Anonymous'}`);
  
  res.json({ 
    success: true, 
    message: 'Appointment booked successfully',
    appointment 
  });
});

// Contact endpoint
app.post('/api/contact', (req, res) => {
  const { name, email, subject, message } = req.body;
  
  if (!name || !email || !message) {
    return res.status(400).json({ error: 'Name, email, and message are required' });
  }
  
  const contact = {
    id: contacts.length + 1,
    name,
    email,
    subject: subject || 'General Inquiry',
    message,
    createdAt: new Date()
  };
  
  contacts.push(contact);
  
  console.log(`Contact form submitted by ${name} (${email}): ${subject}`);
  
  res.json({ 
    success: true, 
    message: 'Thank you for your message. We\'ll get back to you soon!' 
  });
});

// Login endpoint
app.post('/api/login', (req, res) => {
  const { email, password } = req.body;
  
  if (!email || !password) {
    return res.status(400).json({ error: 'Email and password are required' });
  }
  
  // Demo credentials (in real app, this would be validated against a database)
  if (email === 'demo@aura.com' && password === 'demo123') {
    console.log(`User logged in: ${email}`);
    res.json({ 
      success: true, 
      message: 'Login successful',
      token: 'demo-token-' + Date.now(),
      user: { email, name: 'Demo User' }
    });
  } else {
    console.log(`Failed login attempt: ${email}`);
    res.status(401).json({ error: 'Invalid credentials' });
  }
});

// Get chat history
app.get('/api/chat/history', (req, res) => {
  res.json({ 
    success: true, 
    history: chatHistory 
  });
});

// Get appointments
app.get('/api/appointments', (req, res) => {
  res.json({ 
    success: true, 
    appointments 
  });
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    success: true, 
    message: 'Aura Mental Health API is running',
    timestamp: new Date()
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Aura Mental Health API server running on port ${PORT}`);
  console.log(`📡 Health check: http://localhost:${PORT}/api/health`);
  console.log(`💬 Chat endpoint: http://localhost:${PORT}/api/chat`);
  console.log(`📅 Appointments: http://localhost:${PORT}/api/appointments`);
  console.log(`📧 Contact: http://localhost:${PORT}/api/contact`);
  console.log(`🔐 Login: http://localhost:${PORT}/api/login`);
});

