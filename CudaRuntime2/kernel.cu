#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <vector>

#include "Fish.h"
#include "constants.h"
#include "Shader.h"
#include "Parameters.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void processInput(GLFWwindow* window);
int createWindow(GLFWwindow*&);


void setUpParameters();



__global__ void calculatePositionKernel(Fish* fishes,int* dev_indexes,int* dev_headsIndex, float dt, 
	float* vertices, Parameters* parameters) {
	
	int i = threadIdx.x + BLOCK_SIZE * blockIdx.x;
	fishes[i].UpdatePositionKernel(fishes, dev_indexes, dev_headsIndex, dt, parameters);
	fishes[i].SetVertexes(vertices + 12 * i);
}


Parameters parameters;

float increase_step = 0.1;




int main()
{
	GLFWwindow* window;
	int res = createWindow(window);
	if (res == -1)
		return -1;
		
	Shader ourShader("shader.vs", "shader.fs");



	float vertices[NUM_FISH * 12] = { 0 };

	Fish* fishes = new Fish[NUM_FISH];
	for (int i = 0; i < NUM_FISH; i++) {
		int x = rand() % SCR_WIDTH;
		int y = rand() % SCR_HEIGHT;
		fishes[i].SetCordinates((float)x, (float)y);
		if (i<(float)NUM_FISH*0.8)
		{
			fishes[i].SetType(2);
		}
	}
	fishes[0].id = 99;

	Fish* dev_fishes;
	float* dev_vertices;
	int* dev_indexes; // table of fish indexes (hash table) where are indexes sort by mesh
	int* dev_headsIndex; // index of each mesh in indexes table
	Parameters* dev_parameters;

	int indexes[NUM_FISH];
	int headsIndex[NUM_SQUARES];

	setUpParameters();

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudasetdevice failed!  do you have a cuda-capable gpu installed?");
		return 1;
	}

	cudaStatus = cudaMalloc((void**)&dev_vertices, NUM_FISH * sizeof(float) * 12);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_vertices, vertices, NUM_FISH * sizeof(float) * 12, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_fishes, NUM_FISH * sizeof(Fish));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_fishes, fishes, NUM_FISH * sizeof(Fish), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_indexes, NUM_FISH * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_indexes, indexes, NUM_FISH * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_headsIndex, NUM_SQUARES * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_headsIndex, headsIndex, NUM_SQUARES * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_parameters, sizeof(Parameters));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_parameters, &parameters, sizeof(Parameters), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamemcpy failed!");
		goto Error;
	}

	

	glfwSetMouseButtonCallback(window, mouse_button_callback);



	unsigned int  VAO;
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	GLuint VBO;
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), nullptr, GL_DYNAMIC_DRAW);
	cudaGraphicsResource* cudaVBO;
	cudaGraphicsGLRegisterBuffer(&cudaVBO, VBO, cudaGraphicsMapFlagsWriteDiscard);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);


	


	// render loop
	// -----------
	double lastTime = glfwGetTime();
	_sleep(10);
	while (!glfwWindowShouldClose(window))
	{
		// input
		// -----
		processInput(window);

		// render
		// ------
		glClearColor(0.0f, 0.6f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		double currentTime = glfwGetTime();

		size_t num_bytes;

		cudaStatus = cudaGraphicsMapResources(1, &cudaVBO, 0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaGraphicsResourceGetMappedPointer((void**)&dev_vertices, &num_bytes, cudaVBO);

		// making lists for each mesh
		cudaStatus = cudaMemcpy(fishes, dev_fishes, NUM_FISH * sizeof(Fish), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		
		vector<vector<int>> heads(NUM_SQUARES);
		for (int i = 0; i < NUM_SQUARES; i++)
			heads[i] = {};

		for (int i = 0; i < NUM_FISH; i++) {
			int index = Fish::calculateIndexOfMesh(fishes[i].GetX(), fishes[i].GetY());
			heads[index].push_back(i);
		}

		int index = 0;
		for (int i = 0; i < NUM_SQUARES; i++) {
			headsIndex[i] = index;
			for (int j = 0; j < heads[i].size(); j++) {
				indexes[index++] = heads[i][j];
			}
		}

		cudaStatus = cudaMemcpy(dev_indexes, indexes, NUM_FISH * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudamemcpy failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_headsIndex, headsIndex, NUM_SQUARES * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudamemcpy failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_parameters, &parameters, sizeof(Parameters), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudamemcpy failed!");
			goto Error;
		}


		calculatePositionKernel << <NUM_FISH/BLOCK_SIZE, BLOCK_SIZE >> > (dev_fishes, 
			dev_indexes, dev_headsIndex, currentTime - lastTime, dev_vertices, dev_parameters);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "calculatePositionKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		cudaGraphicsUnmapResources(1, &cudaVBO, 0);

		ourShader.use();
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, NUM_FISH * 3);

		glfwSwapBuffers(window);
		glfwPollEvents();

		std::cout << 1 / (currentTime - lastTime) << std::endl;

		lastTime = currentTime;
	}


Error:
	glfwTerminate();
	return 0;
}




void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

int createWindow(GLFWwindow* &window)
{
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// glfw window creation
	// --------------------
	window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		std::cout << "Mouse clicked at position: (" << xpos << ", " << ypos << ")\n";
	}
}

void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
	{
		parameters.stop_simulation = !parameters.stop_simulation;
		cout << "Space pressed" << endl;
	}

	if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
		increase_step = 0.1f;
	if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
		increase_step = 1.0f;
	if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
		increase_step = 10.0f;
	if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
		increase_step = 100.0f;

	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		parameters.avoidWeight -= increase_step;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		parameters.avoidWeight += increase_step;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		parameters.alignWeight -= increase_step;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		parameters.alignWeight += increase_step;
	if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
		parameters.cohesionWeight -= increase_step;
	if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
		parameters.cohesionWeight += increase_step;
	cout << "Avoid: " << parameters.avoidWeight << " Align: "
		<< parameters.alignWeight << " Cohesion: " << parameters.cohesionWeight << endl;
}

void setUpParameters()
{
	parameters.avoidWeight = AVOID_WEIGHT;
	parameters.alignWeight = ALIGN_WEIGHT;
	parameters.cohesionWeight = COHESION_WEIGHT;
	parameters.stop_simulation = STOP_SIMULATION;
	parameters.speed1 = SPEED1;
	parameters.speed2 = SPEED2;
	parameters.maxChangeOfDegreePerSecond1 = MAX_CHANGE_OF_DEGREE_PER_SECOND1;
	parameters.maxChangeOfDegreePerSecond2 = MAX_CHANGE_OF_DEGREE_PER_SECOND2;
	parameters.alignAngle = ALIGN_ANGLE;
	parameters.cohesionAngle = COHESION_ANGLE;
	parameters.avoidAngle = AVOID_ANGLE;
	parameters.avoidDistance = AVOID_DISTANCE;
	parameters.cohesionDistance = COHESION_DISTANCE;
	parameters.alignDistance = ALIGN_DISTANCE;
}