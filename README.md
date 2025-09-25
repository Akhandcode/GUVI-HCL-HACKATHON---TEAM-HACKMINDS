# Aura - Mental Health AI Companion

A full-stack youth mental health web application with a cyber-glow dark theme. Features an AI chat interface, appointment booking, and user authentication.

## Features

- **Cyber-Glow Dark Theme**: Modern, sleek design with neon accents
- **AI Chat Interface**: Always-visible chat with AI companion
- **Appointment Booking**: Modal-based appointment scheduling
- **User Authentication**: Demo login system
- **Responsive Design**: Works on desktop and mobile
- **Real-time Chat**: Interactive AI responses

## Tech Stack

### Frontend
- Html
- Javascript (App.jsx)
- React 19.1.1
- Tailwind CSS 4.1.13
- Inter font family

### Backend
- Node.js with Express
- CORS enabled
- In-memory storage (no database)
- RESTful API endpoints

## Installation & Setup

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start the Backend Server**
   ```bash
   npm run server
   ```
   The server will run on `http://localhost:5000`

3. **Start the Frontend (in a new terminal)**
   ```bash
   npm start
   ```
   The React app will run on `http://localhost:3000`

4. **Or run both simultaneously**
   ```bash
   npm run dev
   ```

## API Endpoints

### Chat
- `POST /api/chat` - Send message to AI
- `GET /api/chat/history` - Get chat history

### Appointments
- `POST /api/appointments` - Book an appointment
- `GET /api/appointments` - Get all appointments

### Contact
- `POST /api/contact` - Submit contact form

### Authentication
- `POST /api/login` - User login

### Health Check
- `GET /api/health` - Server health status

## Demo Credentials

- **Email**: demo@aura.com
- **Password**: demo123

## Design System

### Colors
- **Primary Accent**: Electric Teal (#00F5D4)
- **Secondary Accent**: Bright Magenta (#FF00E6)
- **Background**: Dark Charcoal (#1A1A2E)
- **Component Background**: Dark Navy (#1F1F38)
- **Text Primary**: Off-white (#EAEAEA)
- **Text Secondary**: Soft Gray (#A9A9A9)

### Typography
- **Font Family**: Inter
- **Headings**: Bright off-white
- **Body Text**: Soft gray for readability

## Features in Detail

### Header Component
- Fixed at top with frosted glass effect
- Aura logo with glowing teal accent
- Navigation links with hover effects
- User profile and login button

### Hero Section
- Split-screen layout (60% hero, 40% chat)
- "Find Your Clarity. Anytime." headline
- 2x2 grid of interactive support cards
- Hover effects with glowing borders

### AI Chat Interface
- Always visible and docked on right
- Welcome message from AI companion
- Quick topic buttons
- Message area with user/AI message styling
- Input field with send button
- Quick topic suggestions

### Interactive Elements
- **Book an Appointment**: Opens modal with calendar picker
- **Login/Signup**: Modal-based authentication
- **Quick Topics**: Pre-defined conversation starters
- **Responsive Design**: Mobile-friendly layout

## Development

The application is built as a single-file React component for simplicity, with a separate Express server for the backend. All styling uses Tailwind CSS with custom cyber-glow theme colors.

### File Structure
```
youth-mental-health/
├── src/
│   ├── SimpleApp.jsx    # Main React component (EDIT THIS FILE)
│   ├── App.tsx          # App wrapper
│   ├── App.css          # Additional styles
│   └── App.jsx          # Original complex component
├── server.js            # Express backend (EDIT THIS FILE)
├── package.json         # Dependencies and scripts
└── public/
    └── index.html       # HTML template with Inter font
```

## How to Edit the Application

### 🎨 **Frontend Changes (UI/UX)**
**File to edit:** `src/SimpleApp.jsx`

This is the main file where you can:
- **Add new widgets**: Add more cards in the support cards section (around line 66)
- **Change colors**: Modify the cyber-glow theme colors
- **Add new modals**: Create new modal components for additional features
- **Modify layout**: Change the grid layout, spacing, or component arrangement
- **Update text**: Change headlines, descriptions, and button text
- **Add animations**: Enhance hover effects and transitions

### 🔧 **Backend Changes (API/Data)**
**File to edit:** `server.js`

This is where you can:
- **Add new API endpoints**: Create new routes for additional features
- **Modify existing endpoints**: Change how chat, appointments, etc. work
- **Add database integration**: Replace in-memory storage with real database
- **Add authentication**: Implement proper user login system
- **Add new features**: Create endpoints for meditation, yoga, etc.

### 🎨 **Styling Changes**
**File to edit:** `src/App.css`

For custom CSS that Tailwind can't handle:
- **Custom animations**: Add new keyframe animations
- **Special effects**: Glow effects, particle animations
- **Scrollbar styling**: Custom scrollbar appearance
- **Font imports**: Add new Google Fonts

### 📱 **Responsive Design**
**File to edit:** `src/SimpleApp.jsx`

Modify the Tailwind classes for responsive behavior:
- **Mobile layout**: Change `grid-cols-1 md:grid-cols-2 lg:grid-cols-3`
- **Chat interface**: Adjust `lg:col-span-1` for different screen sizes
- **Modal sizing**: Change `max-w-md` for different modal sizes

## Deployment

For production deployment:

1. Build the React app: `npm run build`
2. Start the server: `npm run server`
3. Serve the built files from the `build` directory

The application is designed to work in a single Replit workspace with both frontend and backend running simultaneously.