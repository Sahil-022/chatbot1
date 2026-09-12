const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const intentMeta = document.getElementById("intent-meta");

function appendMessage(text, sender) {
  const wrapper = document.createElement("div");
  wrapper.classList.add("message", sender);

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.textContent = text;

  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return wrapper;
}

function showTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.classList.add("message", "bot", "typing");
  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.textContent = "Assistant is typing...";
  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return wrapper;
}

async function sendMessage(message) {
  appendMessage(message, "user");
  chatInput.value = "";

  const typingEl = showTypingIndicator();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();

    typingEl.remove();
    appendMessage(data.response, "bot");

    if (data.intent) {
      intentMeta.textContent = `intent: ${data.intent} · confidence: ${(data.confidence * 100).toFixed(0)}%`;
    }
  } catch (err) {
    typingEl.remove();
    appendMessage("Sorry, something went wrong connecting to the server.", "bot");
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  sendMessage(message);
});
