/*
 * Sphere.cpp
 *
 *  Created on: 12 mai 2014
 *      Author: thomas
 */

#include "Sphere.h"
#include <cstdlib>

//Sphere::Sphere(Color color, Position position, float rayon) :
//		m_color(color), m_position(position), m_rayon(rayon) {
//}
//
//Sphere::~Sphere() {
//}
//

Sphere Sphere::random() {
	Sphere sphere;
	int x, y, z;
	x = rand() % 800 * 1000;
	y = rand() % 800 * 1000;
	z = rand() % 800 * 1000 + 1000;
	sphere.m_position.x = x / 1000.f;
	sphere.m_position.y = y / 1000.f;
	sphere.m_position.z = z / 1000.f + 100.f;

	sphere.m_rayon = float(rand() % 50 * 1000 + 50*1000) / 1000.f;

	sphere.m_color.r = rand() % 255;
	sphere.m_color.g = rand() % 255;
	sphere.m_color.b = rand() % 255;

	return sphere;
}
