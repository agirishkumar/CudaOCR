#include "utils.cuh"

std::ofstream Logger::logFile;
LogLevel Logger::currentLevel;

std::string Logger::getTimestamp() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);
    return buf;
}

std::string Logger::getLevelString(LogLevel level) {
    switch (level) {
        case DEBUG: return "DEBUG";
        case INFO: return "INFO";
        case WARNING: return "WARNING";
        case ERROR: return "ERROR";
        case FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

void Logger::init(const std::string& filename, LogLevel level) {
    logFile.open(filename, std::ios::out | std::ios::app);
    currentLevel = level;
}

void Logger::log(LogLevel level, const std::string& message, const char* file, int line) {
    if (level >= currentLevel) {
        std::stringstream ss;
        ss << "[" << getTimestamp() << "] " 
           << "[" << getLevelString(level) << "] "
           << file << ":" << line << " - " 
           << message << std::endl;

        std::cout << ss.str(); 
        logFile << ss.str();   
        logFile.flush();       // Ensures it's written immediately
    }
}

void Logger::close() {
    logFile.close();
}