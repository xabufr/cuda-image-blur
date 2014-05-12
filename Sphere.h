#ifndef SPHERE_H_
#define SPHERE_H_

struct Color {
	int r, g, b;
//	Color(int pr, int pg, int pb) :
//			r(pr), g(pg), b(pb) {
//	}
};
struct Position {
	float x, y, z;
	__device__ float squareLength() const {
		return x * x + y * y + z * z;
	}
//	__device__ __host__ Position(float p_x, float p_y, float p_z = 0) :
//			x(p_x), y(p_y), z(p_z) {
//	}
	__device__ __host__ static Position fromCoord(float p_x, float p_y, float p_z = 0) {
		Position pos;
		pos.x = p_x;
		pos.y = p_y;
		pos.z = p_z;
		return pos;
	}
};
struct Sphere {
public:
//	Sphere(): m_position(0,0,0), m_color(0,0,0) {
//	}
//	Sphere(Color color, Position position, float rayon);
	__device__ float hit(const Position &position,
			float &squareDistanceToCenter) const {
		Position relativeToCenter = Position::fromCoord(0,0,0);
		relativeToCenter.x = m_position.x - position.x;
		relativeToCenter.y = m_position.y - position.y;
		float distToCenter = relativeToCenter.squareLength();
		if (m_rayon * m_rayon >= distToCenter) {
			squareDistanceToCenter = distToCenter;
			return m_position.z - sqrt(m_rayon * m_rayon - distToCenter);
		}
		return -1.f;
	}
	__device__     const Color& color() const {
		return m_color;
	}

	static Sphere random();
	const Position& position() const {
		return m_position;
	}
	__device__ float radius() const {
		return m_rayon;
	}

private:
	float m_rayon;
	Color m_color;
	Position m_position;
};

#endif /* SPHERE_H_ */
