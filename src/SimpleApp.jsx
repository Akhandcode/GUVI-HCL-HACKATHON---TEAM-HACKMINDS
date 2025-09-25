import React, { useState } from 'react';

const SimpleApp = () => {
  const [messages, setMessages] = useState([
    { type: 'ai', text: "I'm Aura, your AI companion. What's on your mind today?", timestamp: new Date() }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [showAppointmentModal, setShowAppointmentModal] = useState(false);
  const [showMeditationModal, setShowMeditationModal] = useState(false);
  const [showYogaModal, setShowYogaModal] = useState(false);
  const [appointmentData, setAppointmentData] = useState({ date: '', time: '', name: '', email: '' });

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = { type: 'user', text: inputMessage, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');

    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputMessage }),
      });

      const data = await response.json();
      
      if (data.success) {
        setMessages(prev => [...prev, { type: 'ai', text: data.response, timestamp: new Date() }]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, { type: 'ai', text: "I'm sorry, I'm having trouble connecting right now.", timestamp: new Date() }]);
    }
  };

  const handleAppointmentSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:5000/api/appointments', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(appointmentData),
      });

      const data = await response.json();
      if (data.success) {
        alert('Appointment booked successfully!');
        setShowAppointmentModal(false);
        setAppointmentData({ date: '', time: '', name: '', email: '' });
      }
    } catch (error) {
      console.error('Error booking appointment:', error);
      alert('Error booking appointment. Please try again.');
    }
  };

  const handleWidgetClick = (widgetType) => {
    switch (widgetType) {
      case 'appointment':
        setShowAppointmentModal(true);
        break;
      case 'meditation':
        setShowMeditationModal(true);
        break;
      case 'yoga':
        setShowYogaModal(true);
        break;
      case 'session':
        // Focus on chat input
        document.querySelector('input[placeholder="Type your message..."]')?.focus();
        break;
      default:
        console.log(`Clicked ${widgetType}`);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
      {/* Header */}
      <header className="bg-gray-800/80 backdrop-blur-md border-b border-cyan-400/20 p-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-cyan-400 to-pink-400 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold">A</span>
            </div>
            <span className="text-2xl font-bold text-white">Aura</span>
          </div>
          <button className="bg-gradient-to-r from-cyan-400 to-pink-400 text-white px-4 py-2 rounded-lg">
            Login
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto p-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Hero Section */}
          <div className="lg:col-span-2">
            <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
              Find Your <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-pink-400">Clarity</span>. Anytime.
            </h1>
            <p className="text-xl text-gray-300 mb-8">
              Aura is your private space to explore your thoughts and find support with an AI companion.
            </p>
            
            {/* Support Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 hover:border-cyan-400/50 hover:shadow-lg hover:shadow-cyan-400/20 transition-all duration-300 transform hover:-translate-y-1 cursor-pointer">
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-10 h-10 bg-cyan-400/20 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white">Consult a Professional</h3>
                </div>
                <p className="text-gray-400 text-sm">Connect with mental health experts</p>
              </div>

              <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 hover:border-cyan-400/50 hover:shadow-lg hover:shadow-cyan-400/20 transition-all duration-300 transform hover:-translate-y-1 cursor-pointer">
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-10 h-10 bg-cyan-400/20 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white">About Our Mission</h3>
                </div>
                <p className="text-gray-400 text-sm">Learn about our approach</p>
              </div>

              <div 
                className="bg-gradient-to-r from-cyan-400/20 to-pink-400/20 backdrop-blur-sm rounded-xl p-6 border border-cyan-400/50 hover:border-pink-400/50 hover:shadow-lg hover:shadow-pink-400/20 transition-all duration-300 transform hover:-translate-y-1 cursor-pointer"
                onClick={() => handleWidgetClick('session')}
              >
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-10 h-10 bg-pink-400/20 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-pink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white">Start a Session</h3>
                </div>
                <p className="text-gray-400 text-sm">Begin your AI conversation</p>
              </div>

              <div 
                className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 hover:border-cyan-400/50 hover:shadow-lg hover:shadow-cyan-400/20 transition-all duration-300 transform hover:-translate-y-1 cursor-pointer"
                onClick={() => handleWidgetClick('appointment')}
              >
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-10 h-10 bg-cyan-400/20 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white">Book Appointment</h3>
                </div>
                <p className="text-gray-400 text-sm">Schedule a session</p>
              </div>

              <div 
                className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 hover:border-pink-400/50 hover:shadow-lg hover:shadow-pink-400/20 transition-all duration-300 transform hover:-translate-y-1 cursor-pointer"
                onClick={() => handleWidgetClick('meditation')}
              >
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-10 h-10 bg-pink-400/20 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-pink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white">Meditation</h3>
                </div>
                <p className="text-gray-400 text-sm">Guided meditation sessions</p>
              </div>

              <div 
                className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 hover:border-green-400/50 hover:shadow-lg hover:shadow-green-400/20 transition-all duration-300 transform hover:-translate-y-1 cursor-pointer"
                onClick={() => handleWidgetClick('yoga')}
              >
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-10 h-10 bg-green-400/20 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white">Yoga & Wellness</h3>
                </div>
                <p className="text-gray-400 text-sm">Mindful movement practices</p>
              </div>
            </div>
          </div>

          {/* Chat Interface */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-cyan-400/20 h-[500px] flex flex-col">
              {/* Chat Header */}
              <div className="p-4 border-b border-gray-700">
                <h3 className="text-lg font-semibold text-white">AI Companion</h3>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((message, index) => (
                  <div key={index} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-xs px-4 py-2 rounded-lg ${
                      message.type === 'user'
                        ? 'bg-gradient-to-r from-pink-400 to-pink-500 text-white'
                        : 'bg-gray-700 text-gray-200'
                    }`}>
                      <p className="text-sm">{message.text}</p>
                    </div>
                  </div>
                ))}
              </div>

              {/* Input */}
              <div className="p-4 border-t border-gray-700">
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Type your message..."
                    className="flex-1 bg-gray-700 text-white placeholder-gray-400 px-4 py-2 rounded-lg border border-gray-600 focus:border-cyan-400 outline-none"
                  />
                  <button
                    onClick={sendMessage}
                    className="bg-gradient-to-r from-cyan-400 to-pink-400 text-white px-4 py-2 rounded-lg hover:from-cyan-500 hover:to-pink-500 transition-all duration-200"
                  >
                    Send
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Appointment Modal */}
      {showAppointmentModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-xl p-6 w-full max-w-md mx-4 border border-cyan-400/20">
            <h3 className="text-xl font-semibold text-white mb-4">Book an Appointment</h3>
            <form onSubmit={handleAppointmentSubmit}>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Date</label>
                  <input
                    type="date"
                    value={appointmentData.date}
                    onChange={(e) => setAppointmentData({ ...appointmentData, date: e.target.value })}
                    className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg border border-gray-600 focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 outline-none"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Time</label>
                  <input
                    type="time"
                    value={appointmentData.time}
                    onChange={(e) => setAppointmentData({ ...appointmentData, time: e.target.value })}
                    className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg border border-gray-600 focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Name</label>
                  <input
                    type="text"
                    value={appointmentData.name}
                    onChange={(e) => setAppointmentData({ ...appointmentData, name: e.target.value })}
                    className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg border border-gray-600 focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
                  <input
                    type="email"
                    value={appointmentData.email}
                    onChange={(e) => setAppointmentData({ ...appointmentData, email: e.target.value })}
                    className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg border border-gray-600 focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 outline-none"
                  />
                </div>
              </div>
              <div className="flex space-x-3 mt-6">
                <button
                  type="button"
                  onClick={() => setShowAppointmentModal(false)}
                  className="flex-1 bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors duration-200"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex-1 bg-gradient-to-r from-cyan-400 to-pink-400 text-white px-4 py-2 rounded-lg hover:from-cyan-500 hover:to-pink-500 transition-all duration-200"
                >
                  Book Appointment
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Meditation Modal */}
      {showMeditationModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-xl p-6 w-full max-w-md mx-4 border border-pink-400/20">
            <h3 className="text-xl font-semibold text-white mb-4">Guided Meditation</h3>
            <div className="space-y-4">
              <div className="bg-gray-700/50 rounded-lg p-4">
                <h4 className="text-lg font-medium text-white mb-2">5-Minute Breathing</h4>
                <p className="text-gray-300 text-sm mb-3">A quick breathing exercise to center yourself</p>
                <button className="bg-pink-400 text-white px-4 py-2 rounded-lg hover:bg-pink-500 transition-colors duration-200">
                  Start Session
                </button>
              </div>
              <div className="bg-gray-700/50 rounded-lg p-4">
                <h4 className="text-lg font-medium text-white mb-2">10-Minute Mindfulness</h4>
                <p className="text-gray-300 text-sm mb-3">A deeper meditation for stress relief</p>
                <button className="bg-pink-400 text-white px-4 py-2 rounded-lg hover:bg-pink-500 transition-colors duration-200">
                  Start Session
                </button>
              </div>
              <div className="bg-gray-700/50 rounded-lg p-4">
                <h4 className="text-lg font-medium text-white mb-2">Sleep Meditation</h4>
                <p className="text-gray-300 text-sm mb-3">Relaxing meditation for better sleep</p>
                <button className="bg-pink-400 text-white px-4 py-2 rounded-lg hover:bg-pink-500 transition-colors duration-200">
                  Start Session
                </button>
              </div>
            </div>
            <button
              onClick={() => setShowMeditationModal(false)}
              className="w-full mt-4 bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors duration-200"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Yoga Modal */}
      {showYogaModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-xl p-6 w-full max-w-md mx-4 border border-green-400/20">
            <h3 className="text-xl font-semibold text-white mb-4">Yoga & Wellness</h3>
            <div className="space-y-4">
              <div className="bg-gray-700/50 rounded-lg p-4">
                <h4 className="text-lg font-medium text-white mb-2">Morning Flow</h4>
                <p className="text-gray-300 text-sm mb-3">Energizing yoga to start your day</p>
                <button className="bg-green-400 text-white px-4 py-2 rounded-lg hover:bg-green-500 transition-colors duration-200">
                  Start Practice
                </button>
              </div>
              <div className="bg-gray-700/50 rounded-lg p-4">
                <h4 className="text-lg font-medium text-white mb-2">Stress Relief</h4>
                <p className="text-gray-300 text-sm mb-3">Gentle poses to release tension</p>
                <button className="bg-green-400 text-white px-4 py-2 rounded-lg hover:bg-green-500 transition-colors duration-200">
                  Start Practice
                </button>
              </div>
              <div className="bg-gray-700/50 rounded-lg p-4">
                <h4 className="text-lg font-medium text-white mb-2">Evening Wind-Down</h4>
                <p className="text-gray-300 text-sm mb-3">Relaxing poses for better sleep</p>
                <button className="bg-green-400 text-white px-4 py-2 rounded-lg hover:bg-green-500 transition-colors duration-200">
                  Start Practice
                </button>
              </div>
            </div>
            <button
              onClick={() => setShowYogaModal(false)}
              className="w-full mt-4 bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors duration-200"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SimpleApp;
