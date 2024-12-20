﻿#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <vector>
#include <fstream>

#include "Fish.h"
#include "constants.h"
#include "Shader.h"
#include "Parameters.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void processInput(GLFWwindow* window);
int createWindow(GLFWwindow*&);


void setUpParameters(int fish_number);


int readNumberOfFishes()
{
	ifstream inputFile("fish_number.txt");
	if (!inputFile) {
		std::cerr << "Error: Could not open the file!" << std::endl;
		return -1;
	}

	int fish_number;

	inputFile >> fish_number;

	if (inputFile.fail()) {
		std::cerr << "Error: Could not read a number from the file!" << std::endl;
		return -1;
	}
	std::cout << "The number read from the file is: " << fish_number << std::endl;
	inputFile.close();
	return fish_number;
}


__global__ void calculatePositionKernel(Fish* fishes,int* dev_indexes,int* dev_headsIndex, float dt, 
	float* vertices, Parameters* parameters) {
	
	int i = threadIdx.x + BLOCK_SIZE * blockIdx.x;
	if (i >= parameters->fish_number)
		return;
	fishes[i].UpdatePositionKernel(fishes, dev_indexes, dev_headsIndex, dt, parameters);
	fishes[i].SetVertexes(vertices + 12 * i);
}


Parameters parameters;

float increase_step = 0.1;

void InitImGui(GLFWwindow* window) {
	// 1. Create ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); // You can access IO for settings
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable keyboard controls (optional)

	// 2. Initialize ImGui backends
	ImGui_ImplGlfw_InitForOpenGL(window, true); // Initialize for GLFW
	ImGui_ImplOpenGL3_Init("#version 330");     // OpenGL version (change to your GLSL version)
}
void RenderImGui(int fps) {
	// Start a new ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	
	ImGui::Begin("Fish settings");
	ImGui::Text("Fps: %d", fps);
	ImGui::SliderFloat("Avoid weight", &parameters.avoidWeight, 0.0f, 5.0f); 
	ImGui::SliderFloat("Align weight", &parameters.alignWeight, 0.0f, 5.0f); 
	ImGui::SliderFloat("Cohesion weight", &parameters.cohesionWeight, 0.0f, 5.0f); 
	ImGui::SliderFloat("Speed1", &parameters.speed1, 10.0f, 500.0f);
	ImGui::SliderFloat("Speed2", &parameters.speed2, 10.0f, 500.0f);
	ImGui::SliderFloat("Max change of degree per second1", &parameters.maxChangeOfDegreePerSecond1, 10.0f, 2000.0f);
	ImGui::SliderFloat("Max change of degree per second2", &parameters.maxChangeOfDegreePerSecond2, 10.0f, 2000.0f);
	if (ImGui::Button(parameters.stop_simulation?"Start simulation":"Stop simulation"))
		parameters.stop_simulation = !parameters.stop_simulation;
	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

int main()
{
	int fish_number = readNumberOfFishes();
	if (fish_number == -1)
	{
		cout << "Podaj liczbe rybek: ";
		cin >> fish_number;
	}
		

	GLFWwindow* window;
	int res = createWindow(window);
	if (res == -1)
		return -1;
	InitImGui(window);
		
	Shader ourShader("shader.vs", "shader.fs");



	float* vertices= new float[fish_number* 12];

	Fish* fishes = new Fish[fish_number];
	for (int i = 0; i < fish_number; i++) {
		int x = rand() % SCR_WIDTH;
		int y = rand() % SCR_HEIGHT;
		fishes[i].SetCordinates((float)x, (float)y);
		if (i<(float)fish_number *0.8)
		{
			fishes[i].SetType(2);
		}
	}

	Fish* dev_fishes;
	float* dev_vertices;
	int* dev_indexes; // table of fish indexes (hash table) where are indexes sort by mesh
	int* dev_headsIndex; // index of each mesh in indexes table
	Parameters* dev_parameters;

	int* indexes= new int[fish_number];
	int headsIndex[NUM_SQUARES];

	setUpParameters(fish_number);

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudasetdevice failed!  do you have a cuda-capable gpu installed?");
		return 1;
	}

	cudaStatus = cudaMalloc((void**)&dev_vertices, fish_number * sizeof(float) * 12);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_vertices, vertices, fish_number * sizeof(float) * 12, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_fishes, fish_number * sizeof(Fish));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_fishes, fishes, fish_number * sizeof(Fish), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_indexes, fish_number * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_indexes, indexes, fish_number * sizeof(int), cudaMemcpyHostToDevice);
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

	

	//glfwSetMouseButtonCallback(window, mouse_button_callback);



	unsigned int  VAO;
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	GLuint VBO;
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*12*fish_number, nullptr, GL_DYNAMIC_DRAW);
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
		cudaStatus = cudaMemcpy(fishes, dev_fishes, fish_number * sizeof(Fish), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		
		vector<vector<int>> heads(NUM_SQUARES);
		for (int i = 0; i < NUM_SQUARES; i++)
			heads[i] = {};

		for (int i = 0; i < fish_number; i++) {
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

		cudaStatus = cudaMemcpy(dev_indexes, indexes, fish_number * sizeof(int), cudaMemcpyHostToDevice);
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

		int numberOfBlocks = ceil(fish_number/(float)BLOCK_SIZE);
		calculatePositionKernel << <numberOfBlocks, BLOCK_SIZE >> > (dev_fishes,
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
		glDrawArrays(GL_TRIANGLES, 0, fish_number * 3);

		RenderImGui(1 / (currentTime - lastTime));

		glfwSwapBuffers(window);
		glfwPollEvents();

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
}

void setUpParameters(int fish_number)
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
	parameters.fish_number = fish_number;
}