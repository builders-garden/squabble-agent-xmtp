import * as fs from "fs";
import {
  AgentKit,
  cdpApiActionProvider,
  cdpWalletActionProvider,
  CdpWalletProvider,
  erc20ActionProvider,
  walletActionProvider,
} from "@coinbase/agentkit";
import { getLangChainTools } from "@coinbase/agentkit-langchain";
import {
  createSigner,
  getEncryptionKeyFromHex,
  logAgentDetails,
  validateEnvironment,
} from "@helpers/client";
import { HumanMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import {
  Client,
  type Conversation,
  type DecodedMessage,
  type XmtpEnv,
} from "@xmtp/node-sdk";
import { setupApiServer } from "./api";
import { createSquabbleTools } from "./lib/tools/squabble-tools";

const {
  WALLET_KEY,
  ENCRYPTION_KEY,
  XMTP_ENV,
  CDP_API_KEY_NAME,
  CDP_API_KEY_PRIVATE_KEY,
  NETWORK_ID,
  OPENAI_API_KEY,
  SQUABBLE_URL,
  AGENT_SECRET,
} = validateEnvironment([
  "WALLET_KEY",
  "ENCRYPTION_KEY",
  "XMTP_ENV",
  "CDP_API_KEY_NAME",
  "CDP_API_KEY_PRIVATE_KEY",
  "NETWORK_ID",
  "OPENAI_API_KEY",
  "SQUABBLE_URL",
  "AGENT_SECRET",
]);

// Storage constants
const XMTP_STORAGE_DIR = ".data/xmtp";
const WALLET_STORAGE_DIR = ".data/wallet";

// Squabble trigger keywords and commands
const SQUABBLE_TRIGGERS = ["@squabble", "@squabble.base.eth"];

// Global stores for memory and agent instances
const memoryStore: Record<string, MemorySaver> = {};
const agentStore: Record<string, Agent> = {};

// Track recent agent messages for context detection
const recentAgentMessages: Record<
  string,
  { timestamp: number; messageId: string }
> = {};
const CONTEXT_WINDOW_MS = 5 * 60 * 1000; // 5 minutes

interface AgentConfig {
  configurable: {
    thread_id: string;
  };
}

type Agent = ReturnType<typeof createReactAgent>;

/**
 * Track when the agent sends a message for context detection
 * @param conversationId - The conversation ID
 * @param messageId - The message ID that was sent
 */
function trackAgentMessage(conversationId: string, messageId: string) {
  recentAgentMessages[conversationId] = {
    timestamp: Date.now(),
    messageId: messageId,
  };
  console.log(`🤖 Tracked agent message in conversation ${conversationId}`);
}

/**
 * Check if a message is within the context window of a recent agent message
 * @param conversationId - The conversation ID
 * @returns boolean - Whether there was a recent agent message
 */
function isWithinContextWindow(conversationId: string): boolean {
  const recentMessage = recentAgentMessages[conversationId];
  if (!recentMessage) {
    return false;
  }

  const timeDiff = Date.now() - recentMessage.timestamp;
  const isWithinWindow = timeDiff <= CONTEXT_WINDOW_MS;

  if (isWithinWindow) {
    console.log(
      `⏰ Message is within context window (${Math.round(timeDiff / 1000)}s ago)`,
    );
  }

  return isWithinWindow;
}

/**
 * Check if a message is a reply to the agent
 * @param message - The decoded XMTP message
 * @param agentInboxId - The agent's inbox ID
 * @returns boolean - Whether the message is a reply to the agent
 */
function isReplyToAgent(
  message: DecodedMessage,
  agentInboxId: string,
): boolean {
  // Check if the message is a reply type
  if (message.contentType?.typeId === "reply") {
    console.log(`📝 Message is a reply type`);
    console.log(`📝 Message ID:`, message.id);
    console.log(`📝 Sender:`, message.senderInboxId);
    console.log(`📝 Content:`, message.content);
    console.log(`📝 ContentType:`, message.contentType);

    // Check additional fields that might contain the reply content
    const messageAny = message as any;
    console.log(`📝 Fallback:`, messageAny.fallback);
    console.log(`📝 Parameters:`, messageAny.parameters);
    console.log(`📝 Compression:`, messageAny.compression);
    console.log(`📝 Kind:`, messageAny.kind);

    return true;
  }
  return false;
}

/**
 * Extract message content from different message types
 * @param message - The decoded XMTP message
 * @returns The message content as a string
 */
function extractMessageContent(message: DecodedMessage): string {
  // Handle reply messages
  if (message.contentType?.typeId === "reply") {
    const messageAny = message as any;
    const replyContent = message.content as any;
    console.log(`🔍 Reply content debug:`, replyContent);

    // Check if content is in the main content field
    if (replyContent && typeof replyContent === "object") {
      // Try different possible property names for the actual content
      if (replyContent.content) {
        return String(replyContent.content);
      }
      if (replyContent.text) {
        return String(replyContent.text);
      }
      if (replyContent.message) {
        return String(replyContent.message);
      }
    }

    // Check fallback field (might contain the actual user message)
    if (messageAny.fallback && typeof messageAny.fallback === "string") {
      console.log(
        `🔍 Found content in fallback field: "${messageAny.fallback}"`,
      );

      // Extract the actual user message from the fallback format
      // Format: 'Replied with "actual message" to an earlier message'
      const fallbackText = messageAny.fallback;
      const match = fallbackText.match(
        /Replied with "(.+)" to an earlier message/,
      );
      if (match && match[1]) {
        const actualMessage = match[1];
        console.log(`🔍 Extracted actual reply content: "${actualMessage}"`);
        return actualMessage;
      }

      // If pattern doesn't match, return the full fallback text
      return fallbackText;
    }

    // Check parameters field (might contain reply data)
    if (messageAny.parameters && typeof messageAny.parameters === "object") {
      const params = messageAny.parameters;
      if (params.content) {
        console.log(
          `🔍 Found content in parameters.content: "${params.content}"`,
        );
        return String(params.content);
      }
      if (params.text) {
        console.log(`🔍 Found content in parameters.text: "${params.text}"`);
        return String(params.text);
      }
    }

    // If content is null/undefined, return empty string to avoid errors
    if (replyContent === null || replyContent === undefined) {
      console.log(
        `⚠️ Reply content is null/undefined, checking other fields failed`,
      );
      return "";
    }

    // Fallback to stringifying the whole content if structure is different
    return JSON.stringify(replyContent);
  }

  // Handle regular text messages
  const content = message.content;
  if (content === null || content === undefined) {
    return "";
  }
  return String(content);
}

/**
 * Check if a message should trigger the Squabble agent
 * @param message - The decoded XMTP message
 * @param agentInboxId - The agent's inbox ID
 * @returns boolean - Whether the agent should respond
 */
function shouldRespondToMessage(
  message: DecodedMessage,
  agentInboxId: string,
): boolean {
  const messageContent = extractMessageContent(message);

  // Safety check for empty content
  if (!messageContent || messageContent.trim() === "") {
    // Special case: if it's a reply type but empty, still check context window
    if (message.contentType?.typeId === "reply") {
      console.log(`⚠️ Empty reply message, checking context window`);
      return isWithinContextWindow(message.conversationId);
    }
    console.log(`⚠️ Empty message content, skipping`);
    return false;
  }

  const lowerMessage = messageContent.toLowerCase().trim();

  // If this is a reply to the agent, always process it
  if (isReplyToAgent(message, agentInboxId)) {
    console.log(`✅ Processing reply to agent: "${messageContent}"`);
    return true;
  }

  // Check if message is within context window of recent agent message
  if (isWithinContextWindow(message.conversationId)) {
    console.log(
      `✅ Processing message within context window: "${messageContent}"`,
    );
    return true;
  }

  // Check if message contains any trigger words/phrases
  const hasTrigger = SQUABBLE_TRIGGERS.some((trigger) =>
    lowerMessage.includes(trigger.toLowerCase()),
  );

  if (hasTrigger) {
    console.log(`✅ Message contains trigger: "${messageContent}"`);
  }

  return hasTrigger;
}

/**
 * Check if message mentions the bot but doesn't use proper triggers
 * @param message - The message content to check
 * @returns boolean - Whether to send a help message
 */
function shouldSendHelpHint(message: string): boolean {
  const lowerMessage = message.toLowerCase().trim();
  const botMentions = ["/bot", "/agent", "/ai", "/help"];

  return (
    botMentions.some((mention) => lowerMessage.includes(mention)) &&
    !SQUABBLE_TRIGGERS.some((trigger) =>
      lowerMessage.includes(trigger.toLowerCase()),
    )
  );
}

/**
 * Ensure local storage directory exists
 */
function ensureLocalStorage() {
  if (!fs.existsSync(XMTP_STORAGE_DIR)) {
    fs.mkdirSync(XMTP_STORAGE_DIR, { recursive: true });
  }
  if (!fs.existsSync(WALLET_STORAGE_DIR)) {
    fs.mkdirSync(WALLET_STORAGE_DIR, { recursive: true });
  }
}

/**
 * Save wallet data to storage.
 *
 * @param userId - The unique identifier for the user
 * @param walletData - The wallet data to be saved
 */
function saveWalletData(userId: string, walletData: string) {
  const localFilePath = `${WALLET_STORAGE_DIR}/${userId}.json`;
  try {
    if (!fs.existsSync(localFilePath)) {
      console.log(`Wallet data saved for user ${userId}`);
      fs.writeFileSync(localFilePath, walletData);
    }
  } catch (error) {
    console.error(`Failed to save wallet data to file: ${error as string}`);
  }
}

/**
 * Get wallet data from storage.
 *
 * @param userId - The unique identifier for the user
 * @returns The wallet data as a string, or null if not found
 */
function getWalletData(userId: string): string | null {
  const localFilePath = `${WALLET_STORAGE_DIR}/${userId}.json`;
  try {
    if (fs.existsSync(localFilePath)) {
      return fs.readFileSync(localFilePath, "utf8");
    }
  } catch (error) {
    console.warn(`Could not read wallet data from file: ${error as string}`);
  }
  return null;
}

/**
 * Initialize the XMTP client.
 *
 * @returns An initialized XMTP Client instance
 */
async function initializeXmtpClient() {
  const signer = createSigner(WALLET_KEY);
  const dbEncryptionKey = getEncryptionKeyFromHex(ENCRYPTION_KEY);

  const identifier = await signer.getIdentifier();
  const address = identifier.identifier;

  const client = await Client.create(signer, {
    dbEncryptionKey,
    env: XMTP_ENV as XmtpEnv,
    dbPath: XMTP_STORAGE_DIR + `/${XMTP_ENV}-${address}`,
  });

  void logAgentDetails(client);

  /* Sync the conversations from the network to update the local db */
  console.log("✓ Syncing conversations...");
  await client.conversations.sync();

  return client;
}

/**
 * Initialize the agent with CDP Agentkit.
 *
 * @param userId - The unique identifier for the user
 * @param conversation - The XMTP conversation instance
 * @param client - The XMTP client instance
 * @param senderWalletAddress - The sender's wallet address
 * @returns The initialized agent and its configuration
 */
async function initializeAgent(
  userId: string,
  conversation: Conversation,
  client: Client,
  senderWalletAddress: string,
): Promise<{ agent: Agent; config: AgentConfig }> {
  try {
    const llm = new ChatOpenAI({
      model: "gpt-4o-mini",
      apiKey: OPENAI_API_KEY,
    });

    const storedWalletData = getWalletData(userId);
    console.log(
      `Wallet data for ${userId}: ${storedWalletData ? "Found" : "Not found"}`,
    );

    const config = {
      apiKeyName: CDP_API_KEY_NAME,
      apiKeyPrivateKey: CDP_API_KEY_PRIVATE_KEY.replace(/\\n/g, "\n"),
      cdpWalletData: storedWalletData || undefined,
      networkId: NETWORK_ID || "base-sepolia",
    };
    console.log(config);

    const walletProvider = await CdpWalletProvider.configureWithWallet(config);

    const agentkit = await AgentKit.from({
      walletProvider,
      actionProviders: [
        walletActionProvider(),
        erc20ActionProvider(),
        cdpApiActionProvider({
          apiKeyName: CDP_API_KEY_NAME,
          apiKeyPrivateKey: CDP_API_KEY_PRIVATE_KEY.replace(/\\n/g, "\n"),
        }),
        cdpWalletActionProvider({
          apiKeyName: CDP_API_KEY_NAME,
          apiKeyPrivateKey: CDP_API_KEY_PRIVATE_KEY.replace(/\\n/g, "\n"),
        }),
      ],
    });

    const tools = await getLangChainTools(agentkit);

    // Add Squabble-specific tools
    const squabbleTools = createSquabbleTools({
      conversation,
      xmtpClient: client,
      senderAddress: userId,
      agentInboxId: client.inboxId,
      squabbleUrl: SQUABBLE_URL,
      agentSecret: AGENT_SECRET,
    });

    const allTools = [...tools, ...squabbleTools];

    console.log(
      `🛠️  Agent initialized with ${tools.length} AgentKit tools + ${squabbleTools.length} Squabble tools = ${allTools.length} total tools`,
    );
    console.log(
      "🎮 Available Squabble tools:",
      squabbleTools.map((tool) => tool.name).join(", "),
    );

    memoryStore[userId] = new MemorySaver();

    const agentConfig: AgentConfig = {
      configurable: { thread_id: userId },
    };

    const agent = createReactAgent({
      llm,
      tools: allTools,
      checkpointSaver: memoryStore[userId],
      messageModifier: `
        "You are a helpful game assistant for Squabble. Keep responses concise and engaging.
        Squabble is a fast-paced, social word game designed for private friend groups on XMTP like the Coinbase Wallet. 
        In each match of 2 to 5 minutes, 2 to 6 players compete on the same randomized letter grid in real-time, racing against the clock to place or create as many words as possible on the grid. 
        The twist? Everyone plays simultaneously on the same board, making every round a shared, high-stakes vocabulary duel.
        The group chat has a leaderboard considering all the matches made on Squabble on that group chat.
        
        IMPORTANT RULES:
        1. When a tool returns a message starting with 'DIRECT_MESSAGE_SENT:', respond with exactly 'TOOL_HANDLED' and nothing else.
        2. When users reply with numbers, amounts, or phrases like 'no bet' after being asked for a bet amount, interpret these as bet amounts and call squabble_start_game with the betAmount parameter.
        3. Examples of bet amount replies: '1', '0.01', 'no bet', '10 $' - all should trigger game creation. The amount must be specificied in $ or USDC or just a number, in the latter case it will be interpreted as USDC. No other tokens!. 
        4. If a user provides what looks like a bet amount (number or 'no bet'), always use the squabble_start_game tool."
      `,
    });

    agentStore[userId] = agent;

    const exportedWallet = await walletProvider.exportWallet();
    const walletDataJson = JSON.stringify(exportedWallet);
    saveWalletData(userId, walletDataJson);

    return { agent, config: agentConfig };
  } catch (error) {
    console.error("Failed to initialize agent:", error);
    throw error;
  }
}

/**
 * Process a message with the agent.
 *
 * @param agent - The agent instance to process the message
 * @param config - The agent configuration
 * @param message - The message to process
 * @returns The processed response as a string
 */
async function processMessage(
  agent: Agent,
  config: AgentConfig,
  message: string,
): Promise<string> {
  let response = "";

  console.log(
    `🤖 Processing message with agent for user: ${config.configurable.thread_id}`,
  );
  console.log(`📝 Message content: "${message}"`);

  try {
    const stream = await agent.stream(
      { messages: [new HumanMessage(message)] },
      config,
    );

    for await (const chunk of stream) {
      if (chunk && typeof chunk === "object" && "agent" in chunk) {
        const agentChunk = chunk as {
          agent: { messages: Array<{ content: unknown }> };
        };
        response += String(agentChunk.agent.messages[0].content) + "\n";
      }
    }

    console.log(`✅ Agent response generated (${response.length} chars)`);
    return response.trim();
  } catch (error) {
    console.error("❌ Error processing message with agent:", error);
    return "Sorry, I encountered an error while processing your request. Please try again later.";
  }
}

/**
 * Handle incoming XMTP messages.
 *
 * @param message - The decoded XMTP message
 * @param client - The XMTP client instance
 */
async function handleMessage(message: DecodedMessage, client: Client) {
  let conversation: Conversation | null = null;
  try {
    const senderAddress = message.senderInboxId;
    const botAddress = client.inboxId.toLowerCase();

    // Ignore messages from the bot itself
    if (senderAddress.toLowerCase() === botAddress) {
      return;
    }

    const messageContent = extractMessageContent(message);
    console.log(`📨 Message from ${senderAddress}: "${messageContent}"`);

    // Get the conversation first
    conversation = (await client.conversations.getConversationById(
      message.conversationId,
    )) as Conversation | null;
    if (!conversation) {
      throw new Error(
        `Could not find conversation for ID: ${message.conversationId}`,
      );
    }

    // Check if message should trigger the Squabble agent
    if (!shouldRespondToMessage(message, client.inboxId)) {
      // Check if they mentioned the bot but didn't use proper triggers
      if (shouldSendHelpHint(messageContent)) {
        await conversation.send(
          "👋 Hi! I'm the Squabble game agent. You asked for help! Try to invoke the agent with @squabble.base.eth or just @squabble\n",
        );
      }
      return;
    }

    // Get the sender's wallet address
    const senderInboxState = await client.preferences.inboxStateFromInboxIds([
      senderAddress,
    ]);
    const senderWalletAddress =
      senderInboxState?.[0]?.recoveryIdentifier?.identifier;

    const { agent, config } = await initializeAgent(
      senderAddress,
      conversation,
      client,
      senderWalletAddress,
    );
    const response = await processMessage(agent, config, messageContent);

    // Don't send "TOOL_HANDLED" responses - these indicate tools have already sent direct messages
    if (response.trim() === "TOOL_HANDLED") {
      console.log(
        "🎮 Tool has already sent direct message - skipping LLM response",
      );
      return;
    }

    const sentMessageId = await conversation.send(response);
    trackAgentMessage(conversation.id, sentMessageId);
    console.log(`✅ Response sent to ${senderAddress}`);
  } catch (error) {
    console.error("❌ Error handling message:", error);
    if (conversation) {
      await conversation.send(
        "I encountered an error while processing your request. Please try again later.",
      );
    }
  }
}

/**
 * Start listening for XMTP messages.
 *
 * @param client - The XMTP client instance
 */
async function startMessageListener(client: Client) {
  console.log("Starting message listener...");
  const messageStream = await client.conversations.streamAllMessages();

  const newGroupStream = client.conversations.stream(
    async (error, conversation) => {
      try {
        if (error) {
          console.log("🔍 Error in conversation stream:", error);
          return;
        }

        if (!conversation) {
          return;
        }

        // Check if this is a new Group (agent was added to a group)
        if (conversation.constructor.name === "Group") {
          //await for 6 seconds
          await new Promise((resolve) => setTimeout(resolve, 6000));
          // Send the message immediately
          conversation
            .send(
              `Squabble is a fast-paced, social word game designed for friend group chats on XMTP. 

In each match, 2 to 6 players compete on the same randomized letter grid in real-time, racing against the clock to place or create as many words as possible on the grid. 

The twist? Everyone plays simultaneously on the same board, making every round a shared, high-stakes vocabulary duel.

The group chat has a leaderboard considering all the matches made on Squabble on that group chat. Use @squabble.base.eth or just @squabble to invoke the squabble agent!`,
            )
            .then(() => {
              console.log("✅ Welcome message sent to new group");
            })
            .catch((error: any) => {
              console.error("❌ Failed to send welcome message:", error);
            });
        }
      } catch (error) {
        console.log("🔍 Error in conversation stream callback:", error);
      }
    },
  );

  console.log("🔍 Conversation stream started with callback");

  // Now the main message stream can run without being blocked
  for await (const message of messageStream) {
    if (message) {
      console.log("🔍 Message received");
      console.log(message.contentType);
      console.log(message.content);
      console.log(message.conversationId);
      console.log(message.kind);
      await handleMessage(message, client);
    }
  }
}

/**
 * Main function to start the chatbot.
 */
async function main(): Promise<void> {
  console.log("Initializing Agent on XMTP...");

  ensureLocalStorage();

  const xmtpClient = await initializeXmtpClient();

  // Start API server
  setupApiServer(xmtpClient);

  // Start XMTP message listener
  await startMessageListener(xmtpClient);
}

// Start the chatbot
main().catch(console.error);
