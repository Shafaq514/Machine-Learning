function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
Diff=((X*Theta')-Y).^2;
J=1/2*sum(sum(R.*Diff))+lambda/2*sum(sum(Theta.^2))+lambda/2*sum(sum(X.^2));

for i=1:num_movies
    R_total=find(R(i,:)==1);         %to find which which user has rated ith movie
    Features=Theta(R_total,:);       %to find the feature vector of users who have rated
    Ratings=Y(i,R_total);                %to find the ratings of ith movie given by users
    X_grad(i,:)=(X(i,:)*Features'-Ratings)*Features+lambda*X(i,:);
end

for i=1:num_users
    Total_R=find(R(:,i));              %to find which movies have been rated by ith user
    Movies_Features=X(Total_R,:);      %to find the features of all the movies rated by ith user
    Ratings=Y(Total_R,i);              %to find the ratings of all the movies by ith user
    Theta_grad(i,:)=(Movies_Features*Theta(i,:)'-Ratings)'*Movies_Features+lambda*Theta(i,:);
end



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
