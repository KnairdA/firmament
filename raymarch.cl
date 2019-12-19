__constant double earth_radius = $earth_radius;
__constant double atmos_height = $atmos_height;

__constant double rayleigh_atmos_height = $rayleigh_atmos_height;
__constant double mie_atmos_height      = $mie_atmos_height;

__constant double3 rayleigh_beta = (double3)$rayleigh_beta;
__constant double3 mie_beta = (double3)$mie_beta;

__constant int ray_samples   = $ray_samples;
__constant int light_samples = $light_samples;

__constant double exposure = $exposure;

bool insideAtmosphere(double3 pos) {
	return length(pos) < earth_radius + atmos_height;
}

double altitude(double3 pos) {
	return length(pos) - earth_radius;
}

// solve x^2 + px + q
bool solvePolynomialOfDegreeTwo(double p, double q, double* x1, double* x2) {
	const double pHalf = 0.5*p;
	const double inner = pHalf*pHalf - q;

	if (inner >= 0.0) {
		*x1 = -pHalf + sqrt(inner);
		*x2 = -pHalf - sqrt(inner);
		return true;
	} else {
		return false;
	}
}

// interection of origin + d*dir and a sphere of radius r for normalized dir
bool solveRaySphereIntersection(double3 origin, double3 dir, double r, double* d0, double* d1) {
	const double p = 2 * dot(dir,origin);
	const double q = dot(origin,origin) - r*r;

	if (solvePolynomialOfDegreeTwo(p, q, d0, d1)) {
		if (*d0 > *d1) {
			double tmp = *d1;
			*d1 = *d0;
			*d0 = tmp;
		}
		return true;
	} else {
		return false;
	}
}

double2 getNormalizedScreenPos(double x, double y) {
	return (double2)(
		2.0 * (x / $size_x - 0.5) * $size_x.0 / $size_y.0,
		2.0 * (y / $size_y - 0.5)
	);
}

double3 getEyeRayDir(double2 screen_pos, double3 eye_pos, double3 eye_target) {
	const double3 forward = normalize(eye_target - eye_pos);
	const double3 right   = normalize(cross((double3)(0.0, 0.0, 1.0), forward));
	const double3 up      = normalize(cross(forward, right));

	return normalize(screen_pos.x*right + screen_pos.y*up + 1.5*forward);
}

bool isVisible(double3 origin, double3 dir) {
	double e0, e1;
	return !solveRaySphereIntersection(origin, dir, earth_radius, &e0, &e1)
	    || (e0 < 0 && e1 < 0);
}

double lengthOfRayInAtmosphere(double3 origin, double3 dir) {
	double d0, d1;
	solveRaySphereIntersection(origin, dir, earth_radius + atmos_height, &d0, &d1);
	return d1 - d0;
}

double3 scatter(double3 origin, double3 dir, double dist, double3 sun) {
	double3 rayleigh_sum = 0.0;
	double3 mie_sum      = 0.0;

	double rayleigh_depth = 0.0;
	double mie_depth      = 0.0;

	const double h = dist / ray_samples;

	for (int i = 0; i < ray_samples; ++i) {
		double3 curr = origin + (i+0.5)*h*dir;

		if (isVisible(curr, sun)) {
			double height = altitude(curr);

			double rayleigh_h = exp(-height / rayleigh_atmos_height) * h;
			double mie_h      = exp(-height / mie_atmos_height) * h;

			rayleigh_depth += rayleigh_h;
			mie_depth      += mie_h;

			double light_h = lengthOfRayInAtmosphere(curr, sun) / light_samples;

			double rayleigh_light_depth = 0;
			double mie_light_depth = 0;

			for (int j = 0; j < light_samples; ++j) {
				height = altitude(curr + (j+0.5)*light_h*sun);

				rayleigh_light_depth += exp(-height / rayleigh_atmos_height) * light_h;
				mie_light_depth += exp(-height / mie_atmos_height) * light_h;
			}

			double3 tau = rayleigh_beta*(rayleigh_depth+rayleigh_light_depth) + mie_beta*(mie_depth+ mie_light_depth);
			double3 attenuation = exp(-tau);

			rayleigh_sum += attenuation * rayleigh_h;
			mie_sum      += attenuation * mie_h;
		}
	}

	const double mu = dot(dir,sun);
	const double rayleigh_phase = 3. / (16.*M_PI) * (1.0 + mu*mu);
	const double g = 0.8;
	const double mie_phase = 3. / (8.*M_PI) * ((1. - g*g) * (1. + mu*mu)) / ((2. + g*g) * pow(1. + g*g - 2.*g*mu, 1.5));

	return rayleigh_sum*rayleigh_beta*rayleigh_phase + mie_sum*mie_beta*mie_phase;
}

void setColor(__global double* result, unsigned x, unsigned y, double3 color) {
	result[3*$size_x*y + 3*x + 0] = color.x;
	result[3*$size_x*y + 3*x + 1] = color.y;
	result[3*$size_x*y + 3*x + 2] = color.z;
}

__kernel void render(__global double* result, double3 eye_pos, double3 eye_dir, double3 sun) {
	eye_pos *= earth_radius;
	eye_dir *= earth_radius;

	const unsigned x = get_global_id(0);
	const unsigned y = get_global_id(1);

	double2 screen_pos = getNormalizedScreenPos(x, y);
	double3 ray_dir = getEyeRayDir(screen_pos, eye_pos, eye_pos + eye_dir);

	double d0, d1;

	if (!solveRaySphereIntersection(eye_pos, ray_dir, earth_radius + atmos_height, &d0, &d1)) {
		setColor(result, x, y, 0.0);
		return;
	}

	double min_dist = d0;
	double max_dist = d1;

	if (insideAtmosphere(eye_pos)) {
		min_dist = 0.0;

		if (solveRaySphereIntersection(eye_pos, ray_dir, earth_radius, &d0, &d1) && d1 > 0) {
			max_dist = max(0.0, d0);
		}
	} else {
		if (solveRaySphereIntersection(eye_pos, ray_dir, earth_radius, &d0, &d1)) {
			max_dist = d0;
		}
	}

	const double3 ray_origin = eye_pos + min_dist*ray_dir;
	const double  ray_length = max_dist - min_dist;

	double3 color = scatter(ray_origin, ray_dir, ray_length, normalize(sun));
	color = 1.0 - exp(-exposure * color);

	setColor(result, x, y, color);
}