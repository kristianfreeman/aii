import { MessageRole, DatabaseMessage } from "@/types";
import Logger from "@/interfaces/logger";

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

export default Memory;