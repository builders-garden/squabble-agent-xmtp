import { validateEnvironment } from "@helpers/client";
import { Client } from "@xmtp/node-sdk";
import express from "express";

const { RECEIVE_AGENT_SECRET } = validateEnvironment([
  "RECEIVE_AGENT_SECRET",
]);

/**
 * Setup Express API server
 * @param client - The XMTP client instance
 */
export function setupApiServer(client: Client) {
  const app = express();
  app.use(express.json());

  // API endpoint to send messages
  app.post("/api/send-message", async (req: any, res: any) => {
    try {
      // Check authentication
      const agentSecret = req.headers["x-agent-secret"];
      const expectedSecret = RECEIVE_AGENT_SECRET;

      if (!expectedSecret) {
        return res.status(500).json({
          error: "Server configuration error: RECEIVE_AGENT_SECRET not set",
        });
      }

      if (!agentSecret || agentSecret !== expectedSecret) {
        return res.status(401).json({
          error: "Unauthorized: Invalid or missing x-agent-secret header",
        });
      }

      const { conversationId, message } = req.body;

      if (!conversationId || !message) {
        return res.status(400).json({
          error: "conversationId and message are required",
        });
      }

      // Get the conversation
      const conversation =
        await client.conversations.getConversationById(conversationId);
      if (!conversation) {
        return res.status(404).json({
          error: "Conversation not found",
        });
      }

      // Send the message
      await conversation.send(message);

      res.json({
        success: true,
        message: "Message sent successfully",
        conversationId,
        sentMessage: message,
      });
    } catch (error) {
      console.error("❌ API Error:", error);
      res.status(500).json({
        error: "Failed to send message",
      });
    }
  });

  const PORT = process.env.PORT || 8080;
  app.listen(PORT, () => {
    console.log(`🚀 API server running on port ${PORT}`);
    console.log(`📡 POST /api/send-message - Send messages to conversations`);
  });

  return app;
}
