const chatEl = document.getElementById("chat");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const statusBadge = document.getElementById("status-badge");
const promptRow = document.getElementById("prompt-row");

let sessionId = crypto.randomUUID();
let isSending = false;

function setStatus(text) {
  statusBadge.textContent = text;
}

function getTimeLabel() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function addBubble(text, role) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = `${role === "user" ? "You" : "Assistant"} • ${getTimeLabel()}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  row.appendChild(meta);
  row.appendChild(bubble);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addTyping() {
  const typing = document.createElement("div");
  typing.className = "typing";
  typing.id = "typing-indicator";
  typing.setAttribute("aria-label", "Assistant is typing");
  typing.innerHTML = "<span class='typing-dot'></span><span class='typing-dot'></span><span class='typing-dot'></span>";
  chatEl.appendChild(typing);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function removeTyping() {
  const typing = document.getElementById("typing-indicator");
  if (typing) typing.remove();
}

function resizeInput() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 140)}px`;
}

async function sendMessage(text) {
  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, user_input: text }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const data = await response.json();
  return data.reply;
}

async function handleSubmit(message) {
  if (isSending) return;
  const text = message.trim();
  if (!text) return;

  isSending = true;
  sendBtn.disabled = true;
  setStatus("Thinking");

  addBubble(text, "user");
  addTyping();

  try {
    const reply = await sendMessage(text);
    removeTyping();
    addBubble(reply, "bot");
    setStatus("Ready");
  } catch (err) {
    removeTyping();
    addBubble("I could not reach the API. Check server/Ollama and try again.", "bot");
    setStatus("Error");
  } finally {
    isSending = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const current = inputEl.value;
  inputEl.value = "";
  resizeInput();
  await handleSubmit(current);
});

inputEl.addEventListener("input", resizeInput);

inputEl.addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    const current = inputEl.value;
    inputEl.value = "";
    resizeInput();
    await handleSubmit(current);
  }
});

clearBtn.addEventListener("click", () => {
  chatEl.innerHTML = "";
  sessionId = crypto.randomUUID();
  addBubble("Session cleared. Ask me anything.", "bot");
  setStatus("New Session");
});

promptRow.addEventListener("click", async (event) => {
  const chip = event.target.closest(".prompt-chip");
  if (!chip) return;
  inputEl.value = chip.textContent.trim();
  resizeInput();
  await handleSubmit(inputEl.value);
  inputEl.value = "";
  resizeInput();
});

addBubble("Hi! I am ready to help. Try one of the prompt chips above.", "bot");
setStatus("Ready");
resizeInput();
