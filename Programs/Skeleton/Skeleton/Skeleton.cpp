//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
#include <time.h>

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	

    //csucspont arnyalo 2 bemeneti regiszterbe var ertelmes adatot                               
    layout(location = 0) in vec3 vp;	 //0 as regiszter

	void main() {
        
		gl_Position = vec4( vp.x , vp.y , vp.z, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";



class Molekul;

vec2 genPoint() { 
	float r, f;
	r = rand() % (600) - 300 + rand() / static_cast<float>(RAND_MAX);
	f = rand() % (600) - 300 + rand() / static_cast<float>(RAND_MAX);
	return vec2(r, f);
}

class Camera2D {
	vec2 wCenter;          
public:
	vec2 wSize;
	Camera2D() :wCenter(0, 0), wSize(600, 600) {}

	mat4 V() { return TranslateMatrix(vec3(-wCenter.x, -wCenter.y, -1)); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }

};

GPUProgram gpuProgram;
Camera2D camera;

float phi;
void Animate(float t) {
	phi = t;
}

class Line {
	unsigned int vao;
	unsigned int vbo;
	vec2  LinePoints[2];
public:

	void create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(
			0,
			2, GL_FLOAT,
			GL_FALSE,
			0,
			0);
	}

	void Draw(vec2 a, vec2 b, vec2 wp, float alpha, vec3 c, mat4 X) {

		LinePoints[0] = a - wp;
		LinePoints[1] = b - wp;

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, c.x, c.y, c.z);



		int x = 0.5;
		int y = 0.5;





		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &X[0][0]);


		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(vec2), &LinePoints[0], GL_STATIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, 2);


	}
};

class Atom {
	unsigned int vao;
	unsigned int vbo;
	vec2 coor;
	vec2 tranCoor;
	vec3 hipCoor;
	int m;
	int q;
	vec2 f;
public:

	void setF(vec2 x) {
		f = x;
	}

	vec2 getF() {
		return f;
	}


	int getQ() {
		return q;
	}

	vec2 getTC() {
		return tranCoor;
	}

	void setQ(int x) {
		q = x;
	}

	vec2 getP() {
		return coor;
	}

	int getM() {
		return m;
	}

	Atom(vec2 x, int c) 
	{
		coor = x;
		tranCoor = coor;
		hipCoor = 0;
		q = c;
		m = rand() % (4)+1;
		
	}


	void create() 
	{

		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
		//glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		
		//glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		

		glEnableVertexAttribArray(0);  
		glVertexAttribPointer(0,       
			3, GL_FLOAT, GL_FALSE, 
			0, 0);
		//printf("a1\n");
	}


	void Draw(vec2 x, float a, vec2 wp, float dt)     //atom rajzolo
	{
		int location = glGetUniformLocation(gpuProgram.getId(), "color");

		if (this->q == 0)
			this->q += 1;

		float qpower = (float)this->q / 5;

		if (this->q > 0) {
			
			glUniform3f(location, qpower, 0, 0);
		}else{ glUniform3f(location, 0, 0, (-1*qpower)); }
		   

		mat4 Y = { cosf(a), -sinf(a), 0, 0,
			       sinf(a), cosf(a), 0, 0,    // row-major!
			       0, 0, 1, 0,
			       0, 0, 0, 1 };


		mat4 MVPtransf =        { 1 , 0, 0, 0,    // MVP matrix, 
								  0,  1, 0, 0,    // row-major!
								  0, 0, 0, 0,
								  0, 0, 0, 1 };

		MVPtransf = MVPtransf * camera.P() * camera.V();

		vec4 buff = vec4(tranCoor.x, tranCoor.y, 0, 0);
		buff = buff - vec4(wp.x,wp.y,0,0);
		buff = buff * Y;
		buff = buff + vec4(x.x, x.y, 0, 0);
		buff = buff + vec4(wp.x, wp.y, 0, 0);
		this->tranCoor = vec2(buff.x, buff.y);


		buff = buff / 300;
		float z = sqrt(buff.x * buff.x + buff.y * buff.y + 1);
		buff = buff / (z + 1);
		buff = buff * 300;
		z = z * 300;

		hipCoor = vec3(buff.x, buff.y,z);





		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);

		
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER ,vbo);

		vec3 vertices[55];                  //55 pontnal a kror mar kornek nez ki
		vec3 v3buff;
		//z = sqrt(buff.x * buff.x + buff.y * buff.y + 1);
		for (int i = 0; i < 55; i++) {
			float fi = ((float)i * 2.0f * (float)M_PI / 55.0f) + ((float)M_PI / 4.0f);
			//vertices[i] = vec2(coor.x+cosf(fi), coor.y+sinf(fi));
			float a = 10*cosf(fi);
			float b = 10*sinf(fi);
			z = sqrt(a * a + b * b + 1);
			v3buff = vec3( a , b, 0 )+tranCoor;
			v3buff = v3buff / 300;
			z = sqrt(v3buff.x * v3buff.x + v3buff.y * v3buff.y + 1);
			v3buff = v3buff / (z + 1);
			v3buff = v3buff * 300;
			vertices[i] = v3buff ;

			/*vertices[i] = vertices[i] / 300;
			z = z / 300;
			vertices[i] = vertices[i] / (z + 1);*/
		}

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(vec3) * 55,
			vertices,
			GL_STATIC_DRAW);

		glDrawArrays(GL_TRIANGLE_FAN, 0, 54 );
		//printf("b\n");
	}

	~Atom() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};



Line l;
float t = 0;

class Molekul {
	std::vector<Atom*> Atoms;
	std::vector<int> Neig;
	unsigned int vao;
	unsigned int vbo;
	vec2 weightP;
	int atomNum;
	vec2 Fse;
	float theta;
	vec3 Mz;
	int m;

	vec2 tesztA ;
	vec2 tesztV;
	vec2 tesztX;

	vec3 e;
	vec3 w;
	vec3 alp;

	std::vector<int> genQinAtoms() {
		int xig;
		std::vector<int> qes;

		if (atomNum % 2 == 1)
			xig = atomNum - 3;
		else
			xig = atomNum;
		//printf("%d",xig);
		int i;
		for (i = 0;i < xig;i++) {
			//printf("%d",i);
			int ran = rand() % (100) + 50;
			qes.push_back(ran);
			i++;
			qes.push_back(-1*ran);
			
		}
		//printf("a0");
		if (atomNum % 2 == 1) {
			qes.push_back(rand() % (1) + 1);
			i++;
			qes.push_back(rand() % (2) + 1);
			i++;
			/*printf("%d\n",-1*((qes.at(i - 2))+(qes.at(i - 1))));
			printf("%d\n", qes.at(i - 2));
			printf("%d\n", qes.at(i - 1));*/
			
			qes.push_back(-1*((qes.at(i - 2))+(qes.at(i - 1)))); //elozo ketto osszege es -1 szerese
		}
		//printf("a0");
		return qes;
	}

	void calWP() {   //calculate WeightPoint
		
		float xbuff = 0;
		float ybuff = 0;
		float mAll = 0;
		for (int i = 0;i < atomNum;i++) {
			xbuff = xbuff + ((float)Atoms.at(i)->getM()* (float)Atoms.at(i)->getTC().x);
			ybuff = ybuff + ((float)Atoms.at(i)->getM()* (float)Atoms.at(i)->getTC().y);
			mAll = mAll + (float)Atoms.at(i)->getM();
		}
		xbuff = xbuff / mAll;
		ybuff = ybuff / mAll;
		weightP = vec2(xbuff, ybuff);

	}


public:

	std::vector<Atom*> getAtoms() {
		return Atoms;
	}

	vec2 getWP() {
		return weightP;
	}
	

	Molekul(int x, int y) {

		tesztA = vec2(0, 0);
		tesztV = vec2(0, 0);
		tesztX = vec2(0, 0);

		vec3 e = vec3(0,0,0);
		vec3 w = vec3(0, 0, 0);
		vec3 alp = vec3(0, 0, 0);

		atomNum = rand()%(6)+2;
		std::vector<int> qValues = genQinAtoms();
		m = 0;
		for (int i = 0; i < atomNum;i++) {
			Atoms.push_back(new Atom(genPoint(),qValues.at(i)));
		}
		for (int i = 0;i < atomNum;i++) {
			m += Atoms.at(i)->getM();
		}
		calWP();

		theta = 0;
		calTheta();

		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,
			3, GL_FLOAT, GL_FALSE,
			0, 0);
	}

	void create() {
		for (int i = 0; i < atomNum;i++) {
			//printf("a");
			//Atoms.at(i)->genFasz();
			Atoms.at(i)->create();
			//printf("%3.2f %3.2f\n",Atoms.at(i)->getP().x, Atoms.at(i)->getP().y);
		}

		TreeNeig();
		
	}

	void calShift() {   //eltolas vektor
		vec2 buff = vec2(0,0);
		for (int i = 0;i < atomNum;i++) {
			vec2 ei = Atoms.at(i)->getTC() - weightP;
			ei = normalize(ei);
			buff = buff + Atoms.at(i)->getF() * ei;
		}
		//printf("%3.2f %3.2f\n", buff.x, buff.y);
		Fse = buff;
		p.Draw(buff*10 + weightP);

	}

	void calTheta() {   //forgatasi tomeg
		for (int i = 0;i < atomNum;i++) {
			float a = length(Atoms.at(i)->getP() - weightP);
			theta = theta + Atoms.at(i)->getM() * (a*a);
		}
		//printf("%3.2f\n", theta);
	}

	void calZDir() {   //z tengely korul
		Mz = vec3(0, 0, 0);
		for (int i = 0;i < atomNum;i++) {
			vec2 a = Atoms.at(i)->getTC();
			vec3 r = vec3(a.x - weightP.x, a.y - weightP.y, 0);
			vec2 b = Atoms.at(i)->getF();
			vec3 f = vec3(b.x, b.y, 0);
			Mz = Mz + cross(r, f);
		}
	}

	void TreeNeig() {
		for (int i = 0;i < atomNum - 1;i++) {
			int r = rand() % (atomNum - (i + 1)) + (i + 1);
			Neig.push_back(i);
			Neig.push_back(r);
		}
	}

	vec3 hip(vec3 buff) {
		buff = buff / 300;
		float z = sqrt(buff.x * buff.x + buff.y * buff.y + 1);
		buff.z = z;
		buff = buff / (z + 1);
		buff = buff * 300;
		return buff;
	}

	bool pin = true;


	

	void Draw(float dt) {
		//printf("%d %d\n",Atoms.size(), Neig.size());
		
		
		if (Atoms.size() > 0) {

			 //p.Draw(this->weightP, weightP);
			
			 
			 calWP();

			 p.Draw(weightP);

			 calShift();
	
			 calZDir();

			 tesztA = Fse / m;
			 tesztV = tesztV +  tesztA * dt;
			 tesztX = tesztV * dt *10 ;

			 

			 e = Mz / theta;
			 w = w + e * dt;
			 alp = w * dt *10 ;

			 //printf("%3.2f\n", alp.z);
			 

			 

			 std::vector<vec3> neigPoints;
			 /*for (int i = 0;i < Neig.size();i++) {
				 vec3 buff = Atoms.at(Neig.at(i))->getTC();
				 neigPoints.push_back(hip(buff));
			 }*/


			 vec3 iv;
			 for (int i = 0;i < Neig.size();) {
				 iv = (Atoms.at(Neig.at(i + 1))->getTC() - Atoms.at(Neig.at(i))->getTC())/10;
				 vec3 buff = Atoms.at(Neig.at(i))->getTC();
				 //printf("%3.2f %3.2f\n", hip(buff).x, hip(buff).y);
				 neigPoints.push_back(hip(buff));

				 /*buff = buff + iv;
				 neigPoints.push_back(hip(buff));*/
				 for (int x = 1;x < 11;x++) {
					 buff = buff + iv;
					 neigPoints.push_back(hip(buff));
					 printf("a");
				 }


				 buff = Atoms.at(Neig.at(i+1))->getTC();
				 neigPoints.push_back(hip(buff));
				 i += 2;

				 glBindVertexArray(vao);
				 glBindBuffer(GL_ARRAY_BUFFER, vbo);

				 int location = glGetUniformLocation(gpuProgram.getId(), "color");
				 glUniform3f(location, 1, 1, 1);

				 glBufferData(GL_ARRAY_BUFFER,
					 sizeof(vec3) * 12,
					 &neigPoints[0],
					 GL_DYNAMIC_DRAW);

				 mat4 MVPtransf = { 1 , 0, 0, 0,    // MVP matrix, 
								  0,  1, 0, 0,    // row-major!
								  0, 0, 0, 0,
								  0, 0, 0, 1 };

				 MVPtransf = MVPtransf * camera.P() * camera.V();


				 location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
				 glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);





				 glDrawArrays(GL_LINE_STRIP, 0, 12);

				 neigPoints.clear();

			 }

			/* glBindVertexArray(vao);
			 glBindBuffer(GL_ARRAY_BUFFER, vbo);

			 int location = glGetUniformLocation(gpuProgram.getId(), "color");
			 glUniform3f(location, 1, 1, 1);

			 glBufferData(GL_ARRAY_BUFFER,
				 sizeof(vec3) * Neig.size(),
				 &neigPoints[0],
				 GL_DYNAMIC_DRAW);

			 mat4 MVPtransf = { 1 , 0, 0, 0,    // MVP matrix, 
								  0,  1, 0, 0,    // row-major!
								  0, 0, 0, 0,
								  0, 0, 0, 1 };

			 MVPtransf = MVPtransf * camera.P() * camera.V();

			 location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
			 glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);

			 glDrawArrays(GL_LINES, 0, 22*Neig.size());*/

			
			for (int i = 0;i < atomNum;i++) {
				Atoms.at(i)->Draw(tesztX, alp.z , weightP, dt);
			}
		}
		
	}

	void Destroy(){
		for (auto p : Atoms)
		{
			delete p;
		}
		Atoms.clear();
		Neig.clear();
		//printf("%d %d\n",Atoms.size(), Neig.size());
	}


};


void calAtomsF(Molekul* m1, Molekul* m2) {

	
	vec2 calP;    //aminek szamolom
	vec2 powP;    //amibol
	vec2 iv;      //iranyvektor
	float dis;    //tavolsag
	float eq;      //ket toltes szorszat
 	float c;      //ero konstans   
	for (int i = 0;i < m1->getAtoms().size();i++) {
		calP = m1->getAtoms().at(i)->getTC();
		for (int x = 0;x < m2->getAtoms().size()-1;x++) {
			eq = m1->getAtoms().at(i)->getQ() * m2->getAtoms().at(x)->getQ()*100;          //100x aranyossag miatt
			powP = m2->getAtoms().at(x)->getTC();
			iv = calP - powP;      
			dis = length(iv);      
			iv = normalize(iv);    //normalizalas 
			c = eq / dis;          
			iv = c * iv;           //erovektor
			
		}
		//p.Draw(iv + calP);
		m1->getAtoms().at(i)->setF(iv);
	}

	for (int i = 0;i < m2->getAtoms().size();i++) {
		calP = m2->getAtoms().at(i)->getTC();
		for (int x = 0;x < m1->getAtoms().size() - 1;x++) {
			eq = m2->getAtoms().at(i)->getQ() * m1->getAtoms().at(x)->getQ() * 100;          //100x aranyossag miatt
			powP = m1->getAtoms().at(x)->getTC();
			iv = calP - powP;
			dis = length(iv);
			iv = normalize(iv);    //normalizalas 
			c = eq / dis;
			iv = c * iv;           //erovektor
		}
		//p.Draw(iv + calP);
		m2->getAtoms().at(i)->setF(iv);
	}
}



/*Atom a1(vec2(0,0),5);
Atom a2(vec2(50,50),-5);*/
Molekul* m1;
Molekul* m2;


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glPointSize(10.0f);
	glLineWidth(1.0f);
	//printf("a0\n");

		

	/*glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
	float vertices[] = { -0.8f, -0.8f, -0.6f, 1.0f, 0.8f, -0.2f };
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vertices),  // # bytes
		vertices,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed*/

	//a1.create(0);
	//a2.create(0);
	m1 = new Molekul(300, 300);
	m1->create();
	m2 = new Molekul(300, 300);
	m2->create();
	l.create();
	p.create();
	//m1.create();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw




void onDisplay() {
	glClearColor(0.5, 0.5, 0.5, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

	//calAtomsF(m1, m2);
	//a1.Draw();

	float dt = phi - t;
	

	//a2.Draw();
	m1->Draw(dt);
	m2->Draw(dt);

	t = phi;
	//l.Draw(m1.Neig.at(0)/300,m1.Neig.at(1)/300);
	calAtomsF(m1, m2);
	//if (pin) {
		//DrawLine();
		//printf("a1");
	//}
	//printf("a1");
	




	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') {
		/*printf("mi a fasz\n");*/
		
		glutPostRedisplay();
	}
	switch (key) {
	case 's': camera.Pan(vec2(-0.1, 0)); break;
	case 'f': camera.Pan(vec2(0.1, 0)); break;
	case 'e': camera.Pan(vec2(0, 0.1)); break;
	case 'x': camera.Pan(vec2(0, -0.1)); break;
	case 'z': camera.Zoom(0.9f); break;
	case 'Z': camera.Zoom(1.1f); break;
	case ' ': 
		m1->Destroy();
		m2->Destroy();
		delete m1;
		delete m2;

		m1 = new Molekul(300,300);
		m1->create();

		printf("a\n");
		m2 = new Molekul(300, 300);
		m2->create();
		printf("a\n");
		break;
	}
	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	/*float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;

	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}*/


}

// Idle event indicating that some time elapsed: do animation here
float sec;

void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;
	Animate(sec);
	glutPostRedisplay();
}

/*mat4 X = { 1, 0, 0, 0,
				   0, 1, 0, 0,    // row-major!
				   0, 0, 0, 0,
				   0, 0, 0, 1 };

X = X * camera.V()*camera.P();*/
