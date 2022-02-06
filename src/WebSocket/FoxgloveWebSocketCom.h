#ifndef JETRACER_WEBSOCKET_H
#define JETRACER_WEBSOCKET_H

#include "../foxglove/websocket/server.h"
#include <nlohmann/json.hpp>
#include <thread>

#include "../EventsThread.h"
#include "../Context.h"
#include "../Events/BaseEvent.h"
#include "../Events/EventTypes.h"

using json = nlohmann::json;

namespace Jetracer
{
    class FoxgloveWebSocketCom : public EventsThread
    {
    public:
        FoxgloveWebSocketCom(const std::string threadName, context_t *ctx);
        // ~FoxgloveWebSocketCom();
    private:
        void handleEvent(pEvent event);
        void Communication();

        context_t *_ctx;
        foxglove::websocket::Server server;
        uint32_t chanId;
        std::thread *CommunicationThread;
    };
} // namespace Jetracer

#endif // JETRACER_WEBSOCKET_H