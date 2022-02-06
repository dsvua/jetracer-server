#include "FoxgloveWebSocketCom.h"
#include "../SlamGpuPipeline/SlamGpuPipeline.h"
#include "../cuda_common.h"
#include <chrono>

// int main()
// {
//     foxglove::websocket::Server server{8765, "example server"};

//     const uint32_t chanId = server.addChannel({
//         .topic = "example_msg",
//         .encoding = "json",
//         .schemaName = "ExampleMsg",
//         .schema =
//             json{
//                 {"type", "object"},
//                 {
//                     "properties", {
//                         {"msg", {{"type", "string"}}},
//                         {"count", {{"type", "number"}}},
//                     },
//                 },
//             }
//                 .dump(),
//     });

//     server.setSubscribeHandler([&](foxglove::websocket::ChannelId chanId)
//                                { std::cout << "first client subscribed to " << chanId << std::endl; });
//     server.setUnsubscribeHandler([&](f#include "../SlamGpuPipeline/SlamGpuPipeline.h"
// oxglove::websocket::ChannelId chanId)
//                                  { std::cout << "last client unsubscribed from " << chanId << std::endl; });

//     uint64_t i = 0;
//     std::shared_ptr<asio::steady_timer> timer;
//     std::function<void()> setTimer = [&]
//     {
//         timer = server.getEndpoint().set_timer(200, [&](std::error_code const &ec)
//                                                {
//       if (ec) {
//         std::cerr << "timer error: " << ec.message() << std::endl;
//         return;
//       }
//       server.sendMessage(chanId, nanosecondsSinceEpoch(),
//                          json{{"msg", "Hello"}, {"count", i++}}.dump());
//       setTimer(); });
//     };

//     setTimer();

//     asio::signal_set signals(server.getEndpoint().get_io_service(), SIGINT);

//     signals.async_wait([&](std::error_code const &ec, int sig)
//                        {
//     if (ec) {
//       std::cerr << "signal error: " << ec.message() << std::endl;
//       return;
//     }
//     std::cerr << "received signal " << sig << ", shutting down" << std::endl;
//     server.removeChannel(chanId);
//     server.stop();
//     if (timer) {
//       timer->cancel();
//     } });

//     server.run();

//     return 0;
// }

namespace Jetracer
{
    static uint64_t nanosecondsSinceEpoch()
    {
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count());
    }

    void normalizeSecNSec(uint64_t &sec, uint64_t &nsec)
    {
        uint64_t nsec_part = nsec % 1000000000UL;
        uint64_t sec_part = nsec / 1000000000UL;

        if (sec + sec_part > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Time is out of dual 32-bit range");

        sec += sec_part;
        nsec = nsec_part;
    }

    // uint64_t sec64 = 0;
    // uint64_t nsec64 = nanosecondsSinceEpoch();
    // normalizeSecNSec(sec64, nsec64);
    // sec = static_cast<uint32_t>(sec64);
    // nsec = static_cast<uint32_t>(nsec64);

    std::string encodeImage(size_t in_len, unsigned char *image)
    {
        static constexpr char sEncodingTable[] = {
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
            'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
            'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
            'w', 'x', 'y', 'z', '0', '1', '2', '3',
            '4', '5', '6', '7', '8', '9', '+', '/'};

        // size_t in_len = data.size();
        size_t out_len = 4 * ((in_len + 2) / 3);
        std::string ret(out_len, '\0');
        size_t i;
        char *p = const_cast<char *>(ret.c_str());

        for (i = 0; i < in_len - 2; i += 3)
        {
            *p++ = sEncodingTable[(image[i] >> 2) & 0x3F];
            *p++ = sEncodingTable[((image[i] & 0x3) << 4) | ((int)(image[i + 1] & 0xF0) >> 4)];
            *p++ = sEncodingTable[((image[i + 1] & 0xF) << 2) | ((int)(image[i + 2] & 0xC0) >> 6)];
            *p++ = sEncodingTable[image[i + 2] & 0x3F];
        }
        if (i < in_len)
        {
            *p++ = sEncodingTable[(image[i] >> 2) & 0x3F];
            if (i == (in_len - 1))
            {
                *p++ = sEncodingTable[((image[i] & 0x3) << 4)];
                *p++ = '=';
            }
            else
            {
                *p++ = sEncodingTable[((image[i] & 0x3) << 4) | ((int)(image[i + 1] & 0xF0) >> 4)];
                *p++ = sEncodingTable[((image[i + 1] & 0xF) << 2)];
            }
            *p++ = '=';
        }

        return ret;
    }

    FoxgloveWebSocketCom::FoxgloveWebSocketCom(const std::string threadName, context_t *ctx) : EventsThread(threadName),
                                                                                               _ctx(ctx),
                                                                                               server(ctx->websocket_server_port, ctx->websocket_server_name)
    {
        auto pushEventCallback = [this](pEvent event) -> bool
        {
            this->pushEvent(event);
            return true;
        };

        _ctx->subscribeForEvent(EventType::event_stop_thread, threadName, pushEventCallback);
        _ctx->subscribeForEvent(EventType::event_gpu_slam_frame, threadName, pushEventCallback);

        // json could be validated at https://jsonlint.com/
        json image_schema = R"({
            "type": "object",
            "properties": {
                "header": {
                    "type": "object",
                    "properties": {
                        "stamp": {
                            "type": "object",
                            "properties": {
                                "sec": {
                                    "type": "integer"
                                },
                                "nsec": {
                                    "type": "integer"
                                }
                            }
                        }
                    }
                },
                "encoding": {
                    "type": "string"
                },
                "data": {
                    "type": "string",
                    "contentEncoding": "base64"
                }
            }
        })"_json;
        // std::cout << "Created image_schema" << std::endl;

        chanId = server.addChannel({
            .topic = "example_image",
            .encoding = "json",
            .schemaName = "ros.sensor_msgs.CompressedImage",
            .schema = image_schema.dump(),
        });

        server.setSubscribeHandler([&](foxglove::websocket::ChannelId chanId)
                                   { std::cout << "first client subscribed to " << chanId << std::endl; });
        server.setUnsubscribeHandler([&](foxglove::websocket::ChannelId chanId)
                                     { std::cout << "last client unsubscribed from " << chanId << std::endl; });

        CommunicationThread = new std::thread(&FoxgloveWebSocketCom::Communication, this);
        // current_send_quota = _ctx->WebSocketCom_max_send_rate;
        std::cout << "FoxgloveWebSocketCom is initialized" << std::endl;
    }

    void FoxgloveWebSocketCom::Communication()
    {

        asio::signal_set signals(server.getEndpoint().get_io_service(), SIGINT);

        signals.async_wait([&](std::error_code const &ec, int sig)
                           {
            if (ec) {
                std::cerr << "signal error: " << ec.message() << std::endl;
                return;
            }
            std::cerr << "received signal " << sig << ", shutting down" << std::endl;
            server.removeChannel(chanId);
            server.stop(); });

        server.run();

        std::cout << "Exiting FoxgloveWebSocket::Communication" << std::endl;
    }

    void FoxgloveWebSocketCom::handleEvent(pEvent event)
    {
        // std::cout << "FoxgloveWebSocketCom::handleEvent Got event of type " << event->event_type << std::endl;

        switch (event->event_type)
        {

        case EventType::event_stop_thread:
        {
            std::cout << "Stopping FoxgloveWebSocketCom Thread" << std::endl;
            server.removeChannel(chanId);
            server.stop();
            CommunicationThread->join();
            std::cout << "Stopped FoxgloveWebSocketCom Thread" << std::endl;
            break;
        }

        case EventType::event_gpu_slam_frame:
        {
            std::shared_ptr<slam_frame_t> slam_frame = std::static_pointer_cast<slam_frame_t>(event->message);

            uint64_t sec64 = 0;
            uint64_t nsec64 = nanosecondsSinceEpoch();
            normalizeSecNSec(sec64, nsec64);
            sec64 = static_cast<uint32_t>(sec64);
            nsec64 = static_cast<uint32_t>(nsec64);

            json message;
            message["encoding"] = "jpeg";
            message["header"]["stamp"]["sec"] = sec64;
            message["header"]["stamp"]["nsec"] = nsec64;
            message["data"] = encodeImage(slam_frame->image_length * sizeof(char), slam_frame->image);
            // std::cout << "Created message" << std::endl;

            // server.sendMessage(chanId, nanosecondsSinceEpoch(), json{{"msg", "Hello"}, {"count", i++}}.dump());
            server.sendMessage(chanId, nanosecondsSinceEpoch(), message.dump());
            break;
        }

        default:
        {
            // std::cout << "Got unknown message of type " << event->event_type << std::endl;
            break;
        }
        }
    }

} // namespace Jetracer
