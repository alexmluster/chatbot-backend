// server.js
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const dotenv = require('dotenv');
const { OpenAI } = require('openai');
const { franc } = require('franc');
const langs = require('langs');

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;
const apikey = process.env.OPENAI_API_KEY;

app.use(cors());
app.use(bodyParser.json());

const userSessions = {};

const openai = new OpenAI({
  apiKey: apikey
});

app.post('/chat', async (req, res) => {
  const { message, userId = 'default-user' } = req.body;

  if (!message) {
    return res.status(400).json({ error: 'Message is required' });
  }

  // ✅ Language detection (inside the POST route)
  const langCode = franc(message);
  const language = langs.where('3', langCode)?.name || 'unknown';

  // ✅ Store context per user
  if (!userSessions[userId]) {
    userSessions[userId] = [];
  }

  userSessions[userId].push({ role: 'user', content: message });

  try {
    const completion = await openai.chat.completions.create({
      messages: userSessions[userId],
      model: 'gpt-4', // or 'gpt-3.5-turbo'
    });

    const reply = completion.choices[0].message.content;

    userSessions[userId].push({ role: 'assistant', content: reply });

    res.json({ reply, language });
  } catch (error) {
    console.error('OpenAI error:', error.response?.data || error.message);
    res.status(500).json({ error: 'Failed to fetch AI response' });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
