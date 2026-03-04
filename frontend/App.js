import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

function App() {
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([]);
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat, loading]);

  const sendMessage = async () => {
    if (!message.trim()) return;

    const userMsg = { sender: "user", text: message };
    setChat(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, history: chat }),
      });
      const data = await response.json();
      setChat(prev => [...prev, { sender: "bot", text: data.response }]);
    } catch {
      setChat(prev => [...prev, { sender: "bot", text: "Server error. Please try again." }]);
    }

    setLoading(false);
    setMessage("");
  };

  return (
    <div className="container">
      <h1>AI Teaching Assistant</h1>

      <div className="chatbox">
        {chat.map((msg, index) => (
          <div key={index} className={msg.sender}>
            <ReactMarkdown>{msg.text}</ReactMarkdown>
          </div>
        ))}

        {loading && (
          <div className="typing">
            <span /><span /><span />
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      <div className="input-area">
        <input
          value={message}
          onChange={e => setMessage(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter") sendMessage(); }}
          placeholder="Ask about the syllabus..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading || !message.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
