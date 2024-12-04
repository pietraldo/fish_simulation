#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <vector>

#include "Fish.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 900;
const unsigned int MESH_SIZE = 100;

const unsigned int NUM_FISH = 2000;
const unsigned int BLOCK_SIZE = 1000;

const int num_rows = SCR_HEIGHT / MESH_SIZE;
const int num_cols = SCR_WIDTH / MESH_SIZE;

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout(location = 1) in float colorId;\n"
"out float vID;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"   vID = colorId;\n"
"}\0";

const char* fragmentShaderSource = "#version 330 core\n"
"in float vID;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"if(vID == 0.0f){\n"
"   FragColor = vec4(1.0f, 0.0f, 0.2f, 1.0f);}\n"
"else if(vID == 1.0f){\n"
"   FragColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);}\n"
"else if(vID == 2.0f){\n"
"   FragColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);}\n"
"}\n\0";


__global__ void calculatePositionKernel(Fish* fishes,int* dev_indexes,int* dev_headsIndex,const int num_squares, float dt, float* vertices, const int n) {
	int i = threadIdx.x + BLOCK_SIZE * blockIdx.x;
	fishes[i].UpdatePositionKernel(fishes, n,  dev_indexes, dev_headsIndex, num_squares, dt, 0, 0, 4000, 50.6, 0.3);
	fishes[i].SetVertexes(vertices + 12 * i);
}

 int calculateIndexOfMesh(float x, float y) {
	int row = y / MESH_SIZE;
	int col = x / MESH_SIZE;
	if (x >= SCR_WIDTH)
		col = num_cols - 1;
	if (y >= SCR_HEIGHT)
		row = num_rows - 1;
	if (x < 0)
		col = 0;
	if (y < 0)
		row = 0;
	return  row * num_cols + col;
}

int mouseX = 0;
int mouseY = 0;
// Mouse button callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		std::cout << "Mouse clicked at position: (" << xpos << ", " << ypos << ")\n";
		mouseX = xpos;
		mouseY = ypos;
	}
}

float avoidWeight = 4000;
float alignWeight = 50.6;
float cohesionWeight = 0.3;

void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		avoidWeight -= 0.1;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		avoidWeight += 0.1;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		alignWeight -= 0.1;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		alignWeight += 0.1;
	if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
		cohesionWeight -= 0.1;
	if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
		cohesionWeight += 0.1;
	//cout << "Avoid: " << avoidWeight << " Align: " << alignWeight << " Cohesion: " << cohesionWeight << endl;
}

struct ListNode {
	Fish* fish;
	int next;
};

int main()
{

	

	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
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




	// compiling vertex shader
	unsigned int vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	int success;
	char infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	unsigned int fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);

	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	// shader program
	unsigned int shaderProgram;
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);



	const int n = NUM_FISH;
	float vertices[n * 12] = { 0 };

	Fish* fishes = new Fish[n];
	for (int i = 0; i < n; i++) {
		int x = rand() % SCR_WIDTH;
		int y = rand() % SCR_HEIGHT;
		fishes[i].SetCordinates((float)x, (float)y);
	}


	Fish* dev_fishes;
	float* dev_vertices;

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudasetdevice failed!  do you have a cuda-capable gpu installed?");
		return 1;
	}


	cudaStatus = cudaMalloc((void**)&dev_vertices, n * sizeof(float) * 12);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_vertices, vertices, n * sizeof(float) * 12, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_fishes, n * sizeof(Fish));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_fishes, fishes, n * sizeof(Fish), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamemcpy failed!");
		goto Error;
	}

	const int num_squares = num_rows * num_cols;
	int indexes[n];
	int headsIndex[num_squares];

	int* dev_indexes;
	int* dev_headsIndex;

	cudaStatus = cudaMalloc((void**)&dev_indexes, n * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_indexes, indexes, n * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_headsIndex, num_squares * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_headsIndex, headsIndex, num_squares * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamemcpy failed!");
		goto Error;
	}

	/*int* dev_heads;
	cudaStatus = cudaMalloc((void**)&dev_heads, n * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}

	ListNode* dev_list;
	cudaStatus = cudaMalloc((void**)&dev_list, n * sizeof(ListNode));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudamalloc failed!");
		goto Error;
	}*/

	glfwSetMouseButtonCallback(window, mouse_button_callback);



	unsigned int  VAO;
	//unsigned int VBO, VAO;
	glGenVertexArrays(1, &VAO);
	//glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	/*glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), nullptr, GL_STATIC_DRAW);*/

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
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
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
		cudaStatus = cudaMemcpy(fishes, dev_fishes, n * sizeof(Fish), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		
		vector<vector<int>> heads(num_squares);
		for (int i = 0; i < num_squares; i++)
			heads[i] = {};

		heads[0].push_back(1);
		for (int i = 0; i < n; i++) {
			int index = calculateIndexOfMesh(fishes[i].GetX(), fishes[i].GetY());
			heads[index].push_back(i);
		}
		
		for (int i = 0; i < num_squares; i++) {
			headsIndex[i] = -1;
		}

		int index = 0;
		for (int i = 0; i < num_squares; i++) {
			for (int j = 0; j < heads[i].size(); j++) {
				if (j == 0)
					headsIndex[i] = j;
				indexes[index++] = heads[i][j];
			}
		}

		cudaStatus = cudaMemcpy(dev_indexes, indexes, n * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudamemcpy failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_headsIndex, headsIndex, num_squares * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudamemcpy failed!");
			goto Error;
		}


		calculatePositionKernel << <NUM_FISH/BLOCK_SIZE, BLOCK_SIZE >> > (dev_fishes, dev_indexes, dev_headsIndex,num_squares, currentTime - lastTime, dev_vertices, n);
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

		/*cudaStatus = cudaMemcpy(vertices, dev_vertices, n * sizeof(float) * 12, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}*/

		/*glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);*/

		cudaGraphicsUnmapResources(1, &cudaVBO, 0);

		glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, n * 3);

		glfwSwapBuffers(window);
		glfwPollEvents();

		std::cout << 1 / (currentTime - lastTime) << std::endl;

		lastTime = currentTime;
	}


Error:
	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();
	return 0;
}



// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}