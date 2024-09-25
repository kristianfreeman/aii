import type { D1Database, KVNamespace, Vectorize } from "@cloudflare/workers-types";
import { generateText } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';

// Type definitions
type MessageRole = "user" | "ai";

interface DatabaseMessage {
  id: number;
  user_id: string;
  message: string;
  role: MessageRole;
  created_at: string;
  summary?: string;
}

// Context interface
interface LLMPContext {
  userId: string;
}

// Logger Interface
interface Logger {
  log(level: string, message: string, meta?: any): void;
  info(message: string, meta?: any): void;
  warn(message: string, meta?: any): void;
  error(message: string, meta?: any): void;
  debug(message: string, meta?: any): void;
}

// Abstract Classes

abstract class Chat {
  protected systemPrompt: string;
  protected logger: Logger;

  constructor(systemPrompt: string, logger: Logger) {
    this.systemPrompt = systemPrompt;
    this.logger = logger;
  }

  abstract generateResponse(
    query: string,
    context: string,
    userPreferences: string,
    facts: string
  ): Promise<string>;
}

abstract class Memory {
  protected userId: string;
  protected logger: Logger;

  constructor(userId: string, logger: Logger) {
    this.userId = userId;
    this.logger = logger;
  }

  abstract getPreviousMessages(limit: number): Promise<string[]>;
  abstract saveMessage(
    message: string,
    role: MessageRole
  ): Promise<number>;
  abstract getMessageById(
    messageId: number
  ): Promise<DatabaseMessage | null>;
}

abstract class Embeddings {
  protected logger: Logger;

  constructor(logger: Logger) {
    this.logger = logger;
  }

  abstract generateEmbeddings(texts: string[]): Promise<number[][]>;
}

abstract class RAG {
  protected userId: string;
  protected logger: Logger;

  constructor(userId: string, logger: Logger) {
    this.userId = userId;
    this.logger = logger;
  }

  abstract retrieveRelevantTexts(
    queryEmbedding: number[],
    topK: number
  ): Promise<string[]>;
  abstract storeEmbedding(
    id: string,
    embedding: number[]
  ): Promise<void>;
}

abstract class Facts {
  protected userId: string;
  protected logger: Logger;

  constructor(userId: string, logger: Logger) {
    this.userId = userId;
    this.logger = logger;
  }

  abstract getFacts(): Promise<string[]>;
  abstract updateFacts(newFact: string): Promise<void>;
  abstract removeFact(fact: string): Promise<void>;
  abstract extractAndUpdateFacts(text: string): Promise<void>;
}

// Cloudflare-specific Implementations

class OpenAIChat extends Chat {
  private openai: any;
  private userId: string;
  private includeDate: boolean;

  constructor(
    systemPrompt: string,
    openaiApiKey: string,
    userId: string,
    logger: Logger,
    includeDate: boolean = true
  ) {
    super(systemPrompt, logger);
    this.openai = createOpenAI({ apiKey: openaiApiKey });
    this.userId = userId;
    this.includeDate = includeDate;
  }

  async generateResponse(
    query: string,
    context: string,
    userPreferences: string,
    facts: string
  ): Promise<string> {
    this.logger.info('Generating AI response', { userId: this.userId });
    const model = this.openai.chat('gpt-4', { user: this.userId });

    let currentDateTimeSection = '';
    if (this.includeDate) {
      const currentDateTime = new Date().toISOString();
      currentDateTimeSection = `<currentDateTime>${currentDateTime}</currentDateTime>`;
    }

    const systemPrompt = `<prompt>${this.systemPrompt.trim()}</prompt>
${currentDateTimeSection}
<userPreferences>${userPreferences.trim()}</userPreferences>
<facts>${facts.trim()}</facts>
<previousConversation>${context.trim()}</previousConversation>`;

    try {
      const resp = await generateText({
        messages: [{ role: 'user', content: query }],
        model,
        system: systemPrompt,
      });
      this.logger.debug('AI response generated successfully');
      return resp.text;
    } catch (error) {
      this.logger.error('Error generating AI response', { error });
      throw new Error('Failed to generate AI response.');
    }
  }
}

class CloudflareMemory extends Memory {
  private db: D1Database;

  constructor(db: D1Database, userId: string, logger: Logger) {
    super(userId, logger);
    this.db = db;
    this.checkTableExists();
  }

  async checkTableExists(): Promise<void> {
    this.logger.debug("Checking if table 'messages' exists in the database...");
    const query = `SELECT 1 FROM messages LIMIT 1`;
    try {
      await this.db.prepare(query).all();
      this.logger.debug("Table 'messages' exists in the database");
    } catch (error: any) {
      if (error && error.message.includes('no such table')) {
        this.logger.error("Table 'messages' does not exist in the database", { error });
        throw new Error("Table 'messages' does not exist in the database.");
      } else {
        this.logger.error('Error checking if table exists', { error });
        throw error;
      }
    }
  }

  async getPreviousMessages(limit: number): Promise<string[]> {
    this.logger.debug('Fetching previous messages', { userId: this.userId, limit });
    const query = `SELECT message FROM messages WHERE user_id = ? ORDER BY created_at DESC LIMIT ?`;
    try {
      const resp = await this.db.prepare(query).bind(this.userId, limit).all();
      const messages = resp.results.map((r: any) => r.message);
      this.logger.debug('Previous messages fetched', { count: messages.length });
      return messages;
    } catch (error) {
      this.logger.error('Error fetching previous messages', { error });
      return [];
    }
  }

  async saveMessage(
    message: string,
    role: MessageRole
  ): Promise<number> {
    this.logger.debug('Saving message', { userId: this.userId, role });
    const insertQuery = `INSERT INTO messages (user_id, message, role) VALUES (?, ?, ?) RETURNING id`;
    try {
      const resp = await this.db
        .prepare(insertQuery)
        .bind(this.userId, message, role)
        .first();
      if (resp && resp.id) {
        this.logger.debug('Message saved', { messageId: resp.id });
        return (resp as any).id;
      } else {
        this.logger.error('Failed to save message');
        throw new Error('Failed to save message.');
      }
    } catch (error) {
      this.logger.error('Error saving message', { error });
      throw error;
    }
  }

  async getMessageById(
    messageId: number
  ): Promise<DatabaseMessage | null> {
    this.logger.debug('Fetching message by ID', { messageId });
    const query = `SELECT * FROM messages WHERE id = ?`;
    try {
      const resp = await this.db.prepare(query).bind(messageId).first<DatabaseMessage>();
      if (resp) {
        this.logger.debug('Message fetched', { messageId });
        return resp;
      } else {
        this.logger.warn('Message not found', { messageId });
        return null;
      }
    } catch (error) {
      this.logger.error('Error fetching message by ID', { error });
      return null;
    }
  }
}

class CloudflareEmbeddings extends Embeddings {
  private ai: any;
  private embeddingsModel: string;

  constructor(
    ai: any,
    logger: Logger,
    embeddingsModel: string = "@cf/baai/bge-base-en-v1.5"
  ) {
    super(logger);
    this.ai = ai;
    this.embeddingsModel = embeddingsModel;
  }

  async generateEmbeddings(texts: string[]): Promise<number[][]> {
    this.logger.debug('Generating embeddings', { textsCount: texts.length });
    try {
      const resp = await this.ai.run(this.embeddingsModel, { text: texts });
      this.logger.debug('Embeddings generated successfully');
      return resp.data;
    } catch (error) {
      this.logger.error('Error generating embeddings', { error });
      throw error;
    }
  }
}

class CloudflareRAG extends RAG {
  private vectorIndex: Vectorize;
  private db: D1Database;

  constructor(
    vectorIndex: Vectorize,
    db: D1Database,
    userId: string,
    logger: Logger
  ) {
    super(userId, logger);
    this.vectorIndex = vectorIndex;
    this.db = db;
  }

  async retrieveRelevantTexts(
    queryEmbedding: number[],
    topK: number
  ): Promise<string[]> {
    this.logger.debug('Retrieving relevant texts using RAG', { userId: this.userId, topK });
    try {
      const resp = await this.vectorIndex.query(queryEmbedding, {
        namespace: `messages:${this.userId}`,
        topK,
      });

      const messageIds = resp.matches.map((match: any) => match.id);
      this.logger.debug('Relevant message IDs retrieved', { messageIds });

      if (messageIds.length === 0) {
        this.logger.debug('No relevant messages found');
        return [];
      }

      // Fetch messages by IDs from the database
      const placeholders = messageIds.map(() => '?').join(',');
      const messagesQuery = `SELECT message FROM messages WHERE id IN (${placeholders})`;
      const messagesResp = await this.db
        .prepare(messagesQuery)
        .bind(...messageIds)
        .all();

      const messages = messagesResp.results.map((r: any) => r.message);
      this.logger.debug('Relevant messages fetched from database', { count: messages.length });
      return messages;
    } catch (error) {
      this.logger.error('Error retrieving relevant texts', { error });
      return [];
    }
  }

  async storeEmbedding(id: string, embedding: number[]): Promise<void> {
    this.logger.debug('Storing embedding in vector index', { id });
    try {
      await this.vectorIndex.upsert([
        {
          id,
          namespace: `messages:${this.userId}`,
          values: embedding,
        },
      ]);
      this.logger.debug('Embedding stored successfully', { id });
    } catch (error) {
      this.logger.error('Error storing embedding', { error });
      throw error;
    }
  }
}

class CloudflareKVFacts extends Facts {
  private kv: KVNamespace;
  private vectorIndex: Vectorize;
  private embeddings: Embeddings;
  private openai: any;

  constructor(
    kv: KVNamespace,
    vectorIndex: Vectorize,
    embeddings: Embeddings,
    openaiApiKey: string,
    userId: string,
    logger: Logger
  ) {
    super(userId, logger);
    this.kv = kv;
    this.vectorIndex = vectorIndex;
    this.embeddings = embeddings;
    this.openai = createOpenAI({ apiKey: openaiApiKey });
  }

  async getFacts(): Promise<string[]> {
    this.logger.debug('Retrieving facts from KV store', { userId: this.userId });
    try {
      const facts = await this.kv.get(`facts:${this.userId}`);
      if (facts) {
        const factsArray = JSON.parse(facts);
        this.logger.debug('Facts retrieved', { count: factsArray.length });
        return factsArray;
      } else {
        this.logger.debug('No facts found');
        return [];
      }
    } catch (error) {
      this.logger.error('Error getting facts', { error });
      return [];
    }
  }

  async updateFacts(newFact: string): Promise<void> {
    this.logger.debug('Updating facts', { userId: this.userId, newFact });
    try {
      // Generate embeddings for the new fact
      const [newFactEmbedding] = await this.embeddings.generateEmbeddings([newFact]);

      // Check if the fact already exists using Vectorize
      const resp = await this.vectorIndex.query(newFactEmbedding, {
        namespace: `facts:${this.userId}`,
        topK: 1,
      });

      const exists =
        resp.matches &&
        resp.matches.length > 0 &&
        resp.matches[0].score > 0.9;

      if (exists) {
        // Fact already exists, maybe update it
        await this.removeFact(resp.matches[0].id);
        this.logger.debug('Existing fact removed', { factId: resp.matches[0].id });
      }

      // Store the fact in KV
      let facts = await this.getFacts();
      facts.push(newFact);
      await this.kv.put(`facts:${this.userId}`, JSON.stringify(facts));
      this.logger.debug('New fact stored in KV', { fact: newFact });

      // Store the embedding in Vectorize
      await this.vectorIndex.upsert([
        {
          id: newFact,
          namespace: `facts:${this.userId}`,
          values: newFactEmbedding,
        },
      ]);
      this.logger.debug('New fact embedding stored in Vectorize', { fact: newFact });
    } catch (error) {
      this.logger.error('Error updating facts', { error });
      throw error;
    }
  }

  async removeFact(fact: string): Promise<void> {
    this.logger.debug('Removing fact', { userId: this.userId, fact });
    try {
      // Remove from KV
      let facts = await this.getFacts();
      facts = facts.filter((f) => f !== fact);
      await this.kv.put(`facts:${this.userId}`, JSON.stringify(facts));
      this.logger.debug('Fact removed from KV', { fact });

      // Remove from Vectorize
      const vectors = await this.embeddings.generateEmbeddings([fact]);
      const vector = vectors[0];
      const embedding = await this.vectorIndex.query(vector, {
        namespace: `facts:${this.userId}`,
        topK: 1,
      });
      const match = embedding.matches[0];
      await this.vectorIndex.deleteByIds([match.id]);
      this.logger.debug('Fact removed from Vectorize', { fact });
    } catch (error) {
      this.logger.error('Error removing fact', { error });
      throw error;
    }
  }

  async extractAndUpdateFacts(text: string): Promise<void> {
    this.logger.debug('Extracting facts from text');
    const model = this.openai.chat('gpt-4', { user: this.userId });

    const systemPrompt = `Extract any new facts or information from the following text and present them as a list of concise statements. If there are no new facts, return an empty list.`;

    try {
      const resp = await generateText({
        messages: [{ role: 'user', content: text }],
        model,
        system: systemPrompt,
      });

      const extractedFacts = resp.text
        .split('\n')
        .map((fact) => fact.trim())
        .filter((fact) => fact);

      this.logger.debug('Facts extracted', { count: extractedFacts.length });

      for (const fact of extractedFacts) {
        await this.updateFacts(fact);
        this.logger.debug('Fact updated', { fact });
      }
    } catch (error) {
      this.logger.error('Error extracting and updating facts', { error });
    }
  }
}

// LLMP Class

class LLMP {
  private chat: Chat;
  private memory: Memory;
  private rag: RAG;
  private facts: Facts;
  private embeddings: Embeddings;
  private context: LLMPContext;
  private logger: Logger;

  constructor({
    chat,
    memory,
    rag,
    facts,
    embeddings,
    context,
    logger,
  }: {
    chat: Chat;
    memory: Memory;
    rag: RAG;
    facts: Facts;
    embeddings: Embeddings;
    context: LLMPContext;
    logger: Logger;
  }) {
    this.chat = chat;
    this.memory = memory;
    this.rag = rag;
    this.facts = facts;
    this.embeddings = embeddings;
    this.context = context;
    this.logger = logger;
  }

  async handleQuery(
    query: string,
    userPreferences: string
  ): Promise<string> {
    this.logger.info('Received user query', { userId: this.context.userId, query });

    // Save user message
    const userMessageId = await this.memory.saveMessage(query, "user");

    // Generate embeddings for the user message
    this.logger.debug('Generating embeddings for user message', { messageId: userMessageId });
    const [userMessageEmbedding] = await this.embeddings.generateEmbeddings([
      query,
    ]);

    // Store the embedding in the vector store
    await this.rag.storeEmbedding(
      userMessageId.toString(),
      userMessageEmbedding
    );
    this.logger.debug('Stored user message embedding', { messageId: userMessageId });

    // Get previous messages for context
    const previousMessages = await this.memory.getPreviousMessages(10);
    this.logger.debug('Retrieved previous messages for context', { count: previousMessages.length });
    const context = previousMessages.join('\n');

    // Retrieve relevant texts using RAG
    const relevantTexts = await this.rag.retrieveRelevantTexts(
      userMessageEmbedding,
      5
    );
    this.logger.debug('Retrieved relevant texts using RAG', { count: relevantTexts.length });
    const relevantContext = relevantTexts.join('\n');

    // Get facts
    const factsArray = await this.facts.getFacts();
    this.logger.debug('Retrieved facts', { count: factsArray.length });
    const facts = factsArray.join('\n');

    // Generate AI response
    const aiResponse = await this.chat.generateResponse(
      query,
      context + '\n' + relevantContext,
      userPreferences,
      facts
    );
    this.logger.info('Generated AI response');

    // We intentionally do not save AI message or generate embeddings for it
    // We also do not extract new facts from it
    // This may change in the future

    // Return the AI response
    return aiResponse;
  }
}

// Exporting classes and LLMP for external use
export {
  Logger,
  Chat,
  Memory,
  Embeddings,
  RAG,
  Facts,
  OpenAIChat,
  CloudflareMemory,
  CloudflareEmbeddings,
  CloudflareRAG,
  CloudflareKVFacts,
  LLMP,
};
