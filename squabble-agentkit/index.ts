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
  NEYNAR_API_KEY,
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
  "NEYNAR_API_KEY",
]);

// Storage constants
const XMTP_STORAGE_DIR = ".data/xmtp";
const WALLET_STORAGE_DIR = ".data/wallet";

// Squabble trigger keywords and commands
const SQUABBLE_TRIGGERS = ["/squabble"];

// Conversation state management
interface ConversationState {
  isWaitingForResponse: boolean;
  lastCommand: string;
  messageCount: number;
  lastUpdate: number;
}

const STATE_TIMEOUT = 60 * 1000; // 1 minute in milliseconds
const MAX_CONTEXT_MESSAGES = 5; // Reduced from 10 to 5 for general context

// Global stores for memory, agent instances, and conversation states
const memoryStore: Record<string, MemorySaver> = {};
const agentStore: Record<string, Agent> = {};
const conversationStates: Record<string, ConversationState> = {};

interface AgentConfig {
  configurable: {
    thread_id: string;
  };
}

type Agent = ReturnType<typeof createReactAgent>;

/**
 * Get conversation state with automatic cleanup
 * @param conversationId - The conversation ID
 * @returns The current conversation state
 */
function getConversationState(conversationId: string): ConversationState {
  const state = conversationStates[conversationId];
  const now = Date.now();

  // Clear state if it's expired
  if (!state) {
    const newState = {
      isWaitingForResponse: false,
      lastCommand: "",
      messageCount: 0,
      lastUpdate: now,
    };
    conversationStates[conversationId] = newState;
    return newState;
  }

  // Check if state is expired
  const timeDiff = now - state.lastUpdate;
  if (timeDiff > STATE_TIMEOUT) {
    console.log(`⏰ State expired, resetting`);
    const newState = {
      isWaitingForResponse: false,
      lastCommand: "",
      messageCount: 0,
      lastUpdate: now,
    };
    conversationStates[conversationId] = newState;
    return newState;
  }

  // Check if message count exceeded
  if (state.messageCount >= MAX_CONTEXT_MESSAGES) {
    console.log(`📊 Message count exceeded, resetting`);
    const newState = {
      isWaitingForResponse: false,
      lastCommand: "",
      messageCount: 0,
      lastUpdate: now,
    };
    conversationStates[conversationId] = newState;
    return newState;
  }

  return state;
}

/**
 * Update conversation state
 * @param conversationId - The conversation ID
 * @param isWaitingForResponse - Whether we're waiting for a response
 */
function updateConversationState(
  conversationId: string,
  isWaitingForResponse: boolean,
) {
  const currentState = conversationStates[conversationId] || {
    isWaitingForResponse: false,
    lastCommand: "",
    messageCount: 0,
    lastUpdate: Date.now(),
  };

  conversationStates[conversationId] = {
    isWaitingForResponse,
    lastCommand: currentState.lastCommand,
    messageCount: currentState.messageCount + 1,
    lastUpdate: Date.now(),
  };
}

/**
 * Set conversation state with a specific command
 * @param conversationId - The conversation ID
 * @param isWaitingForResponse - Whether we're waiting for a response
 * @param command - The command that initiated this state
 */
function setConversationState(
  conversationId: string,
  isWaitingForResponse: boolean,
  command: string = "",
) {
  const currentState = conversationStates[conversationId] || {
    isWaitingForResponse: false,
    lastCommand: "",
    messageCount: 0,
    lastUpdate: Date.now(),
  };

  conversationStates[conversationId] = {
    isWaitingForResponse,
    lastCommand: command,
    messageCount: currentState.messageCount + 1,
    lastUpdate: Date.now(),
  };
}

/**
 * Check if a message should trigger the Squabble agent
 * @param message - The message content to check
 * @param conversationId - The conversation ID to check state for
 * @returns boolean - Whether the agent should respond
 */
function shouldRespondToMessage(
  message: string,
  conversationId: string,
): boolean {
  const lowerMessage = message.toLowerCase().trim();
  const state = getConversationState(conversationId);

  // If we're waiting for a response, process any message
  if (state.isWaitingForResponse) {
    console.log(`✅ Processing message in waiting state: "${message}"`);
    return true;
  }

  // Check if message contains any trigger words/phrases
  const hasTriger = SQUABBLE_TRIGGERS.some((trigger) =>
    lowerMessage.includes(trigger.toLowerCase()),
  );

  if (hasTriger) {
    console.log(`✅ Message contains trigger: "${message}"`);
  }

  return hasTriger;
}

/**
 * Check if message mentions the bot but doesn't use proper triggers
 * @param message - The message content to check
 * @returns boolean - Whether to send a help message
 */
function shouldSendHelpHint(message: string): boolean {
  const lowerMessage = message.toLowerCase().trim();
  const botMentions = ["bot", "agent", "ai", "help"];

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
        The group chat has a leaderboard considering all the matches made on Squabble on that group chat."
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

    const messageContent = String(message.content);
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

    const state = getConversationState(conversation.id);

    // Check if we're waiting for a response
    if (state.isWaitingForResponse) {
      console.log(`📝 Processing response in waiting state`);

      // Check if user typed a new trigger command - if so, reset state and process normally
      const hasNewTrigger = SQUABBLE_TRIGGERS.some((trigger) =>
        messageContent.toLowerCase().includes(trigger.toLowerCase()),
      );

      if (hasNewTrigger) {
        console.log(`🔄 New trigger detected - resetting state`);
        updateConversationState(conversation.id, false);
        // Continue processing the new command normally (don't return here)
      } else {
        // Get the sender's wallet address
        const senderInboxState =
          await client.preferences.inboxStateFromInboxIds([senderAddress]);
        const senderWalletAddress =
          senderInboxState?.[0]?.recoveryIdentifier?.identifier;

        // Initialize agent and process the response
        const { agent, config } = await initializeAgent(
          senderAddress,
          conversation,
          client,
          senderWalletAddress,
        );

        const response = await processMessage(agent, config, messageContent);

        // Always set waiting state after processing any trigger to maintain context
        setConversationState(conversation.id, true, messageContent);

        await conversation.send(response);
        console.log(`✅ Response sent to ${senderAddress}`);
        return;
      }
    }

    // Check if message should trigger the Squabble agent
    if (!shouldRespondToMessage(messageContent, conversation.id)) {
      // Check if they mentioned the bot but didn't use proper triggers
      if (shouldSendHelpHint(messageContent)) {
        await conversation.send(
          "👋 Hi! I'm the Squabble game bot. Try using:\n" +
            "• `/squabble help` - Get game rules\n" +
            "• `/squabble start` - Create a new game\n" +
            "• `/squabble leaderboard` - View rankings\n" +
            "• Or just say 'start game', 'show leaderboard', etc.",
        );
      }
      return;
    }

    // Check if this is a start game command that should prompt for bet
    const lowerMessage = messageContent.toLowerCase();
    if (
      lowerMessage.includes("/squabble start") &&
      !lowerMessage.includes("bet")
    ) {
      setConversationState(conversation.id, true, "/squabble start");
      await conversation.send(
        "🎮 How much would you like to bet for this game? You can enter an amount or say 'no bet' if you prefer.",
      );
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

    // Always set waiting state after processing any trigger to maintain context
    setConversationState(conversation.id, true, messageContent);

    await conversation.send(response);
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

  const newGroupStream = client.conversations.stream((error, conversation) => {
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
        // Send the message immediately
        conversation
          .send(
            `Squabble is a fast-paced, social word game designed for friend group chats on XMTP. 

In each match, 2 to 6 players compete on the same randomized letter grid in real-time, racing against the clock to place or create as many words as possible on the grid. 

The twist? Everyone plays simultaneously on the same board, making every round a shared, high-stakes vocabulary duel.

The group chat has a leaderboard considering all the matches made on Squabble on that group chat. Use /squabble to invoke the squabble agent!`,
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
  });

  console.log("🔍 Conversation stream started with callback");

  // Now the main message stream can run without being blocked
  for await (const message of messageStream) {
    if (message) {
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
  await startMessageListener(xmtpClient);
}

// Start the chatbot
main().catch(console.error);
