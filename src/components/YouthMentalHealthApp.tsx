import React, { useState } from "react";

const YouthMentalHealthApp: React.FC = () => {
  const [currentView, setCurrentView] = useState<"home" | "contact" | "resources">("home");
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState<string | null>(null);

  const [contactName, setContactName] = useState("");
  const [contactEmail, setContactEmail] = useState("");
  const [contactMessage, setContactMessage] = useState("");

  const suggestedTopics = ["Stress Management", "Mindfulness", "Peer Support", "Healthy Habits"];

  // Handlers
  const handleSendMessage = () => {
    alert(`Message sent!\nName: ${contactName}\nEmail: ${contactEmail}\nMessage: ${contactMessage}`);
    setContactName("");
    setContactEmail("");
    setContactMessage("");
  };

  const handleLogin = () => {
    setIsLoggedIn(true);
    setUser("John Doe");
  };

  const handleSignup = () => {
    alert("Signup successful!");
    setIsLoggedIn(true);
    setUser("New User");
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUser(null);
  };

  return (
    <div className="app">
      <header>
        <h1>Youth Mental Health App</h1>
        <nav>
          <button onClick={() => setCurrentView("home")}>Home</button>
          <button onClick={() => setCurrentView("resources")}>Resources</button>
          <button onClick={() => setCurrentView("contact")}>Contact</button>
        </nav>
        <div>
          {isLoggedIn ? (
            <>
              <span>Welcome, {user}!</span>
              <button onClick={handleLogout}>Logout</button>
            </>
          ) : (
            <>
              <button onClick={handleLogin}>Login</button>
              <button onClick={handleSignup}>Signup</button>
            </>
          )}
        </div>
      </header>

      <main>
        {currentView === "home" && (
          <section>
            <h2>Home</h2>
            <p>Welcome to our safe space for youth mental health.</p>
          </section>
        )}

        {currentView === "resources" && (
          <section>
            <h2>Suggested Topics</h2>
            <ul>
              {suggestedTopics.map((topic, index) => (
                <li key={index}>{topic}</li>
              ))}
            </ul>
          </section>
        )}

        {currentView === "contact" && (
          <section>
            <h2>Contact Us</h2>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage();
              }}
            >
              <input
                type="text"
                placeholder="Your Name"
                value={contactName}
                onChange={(e) => setContactName(e.target.value)}
              />
              <input
                type="email"
                placeholder="Your Email"
                value={contactEmail}
                onChange={(e) => setContactEmail(e.target.value)}
              />
              <textarea
                placeholder="Your Message"
                value={contactMessage}
                onChange={(e) => setContactMessage(e.target.value)}
              />
              <button type="submit">Send Message</button>
            </form>
          </section>
        )}
      </main>

      <footer>
        <p>© 2025 Youth Mental Health Project</p>
      </footer>
    </div>
  );
};

export default YouthMentalHealthApp;
