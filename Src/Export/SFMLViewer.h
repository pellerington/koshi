#pragma once

#include <SFML/Graphics.hpp>
#include <iostream>

#include "../Render/Render.h"

class SFMLViewer
{
public:
    static bool RenderWindow(int screen_height, Render * render)
    {
        int image_width = render->get_image_resolution()[0];
        int image_height = render->get_image_resolution()[1];
        int screen_width = screen_height * (float)image_width / image_height;
        sf::RenderWindow window(sf::VideoMode(screen_width, screen_height), "My window");

        sf::Texture texture;
        texture.create(image_width, image_height);
        sf::Uint8 pixels[image_width * image_height * 4];
        sf::Sprite sprite(texture);

        sprite.setScale((float)screen_width/image_width, (float)screen_height/image_height);

        while (window.isOpen())
        {
            sf::Event event;
            while (window.pollEvent(event))
            {
                if (event.type == sf::Event::Closed)
                {
                    sf::Image output;
                    output.create(image_width, image_height, pixels);
                    output.saveToFile("output.png");
                    window.close();
                }
            }

            update_pixels(render, pixels);
            texture.update(pixels);
            window.clear(sf::Color::Black);
            window.draw(sprite);
            window.display();

            sf::sleep(sf::milliseconds(500));
        }
        return true;
    }

private:
    static void update_pixels(Render * render, sf::Uint8 * pixels)
    {
        int image_width = render->get_image_resolution()[0];
        int image_height = render->get_image_resolution()[1];

        for(int x = 0; x < image_width; x++)
        {
            for(int y = 0; y < image_height; y++)
            {
                Vec3f pixel = render->get_pixel(x, y);
                int index = (x + image_width * y) * 4;
                pixels[index + 0] = std::min(std::max((int)(pixel[0] * 255), 0), 255);
                pixels[index + 1] = std::min(std::max((int)(pixel[1] * 255), 0), 255);
                pixels[index + 2] = std::min(std::max((int)(pixel[2] * 255), 0), 255);
                pixels[index + 3] = 255;
            }
        }
    }
};
