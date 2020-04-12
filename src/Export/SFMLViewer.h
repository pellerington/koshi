#pragma once

#include <SFML/Graphics.hpp>
#include <iostream>

#include <Render/Render.h>

class SFMLViewer
{
public:
    static bool RenderWindow(const uint screen_height, const Render * render)
    {
        const uint image_width = render->get_image_resolution().x;
        const uint image_height = render->get_image_resolution().y;
        const uint screen_width = screen_height * (float)image_width / image_height;
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
                    window.close();
                if (event.type == sf::Event::MouseButtonPressed)
                    if (event.mouseButton.button == sf::Mouse::Left)
                        std::cout << render->get_pixel_color(image_width * event.mouseButton.x/screen_width, image_height * event.mouseButton.y/screen_height) << std::endl;
            }

            update_pixels(render, pixels);
            texture.update(pixels);
            window.clear(sf::Color::Black);
            window.draw(sprite);
            window.display();

            sf::sleep(sf::milliseconds(750));
        }
        return true;
    }

private:
    static void update_pixels(const Render * render, sf::Uint8 * pixels)
    {
        uint image_width = render->get_image_resolution().x;
        uint image_height = render->get_image_resolution().y;

        for(uint x = 0; x < image_width; x++)
        {
            for(uint y = 0; y < image_height; y++)
            {
                const Vec3f pixel = render->get_pixel_color(x, y);
                const uint index = (x + image_width * y) * 4;
                pixels[index + 0] = std::min(std::max((int)(pixel.r * 255), 0), 255);
                pixels[index + 1] = std::min(std::max((int)(pixel.g * 255), 0), 255);
                pixels[index + 2] = std::min(std::max((int)(pixel.b * 255), 0), 255);
                pixels[index + 3] = 255; //alpha
            }
        }
    }
};
